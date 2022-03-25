import functools
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Callable, DefaultDict, List, Optional, Set, Tuple, cast

import base_main
import gym
import habitat
import heatmap
import line_chart
import numpy as np
from envs import VecPyTorch
from my import env
from my.agent import Agent
from my.env import PAIR, Env, Mesh
from run_logger import HasuraLogger
from stable_baselines3.common.monitor import Monitor
from transformers import CLIPProcessor, GPT2Tokenizer
from wrappers import (
    EPISODE_SUCCESS,
    ActionSpaceWrapper,
    CLIPProcessorWrapper,
    GPT3Wrapper,
    MissionPreprocessor,
    ObsWrapper,
    RenderWrapper,
    RolloutsWrapper,
    TokenizerWrapper,
    TrainTest,
)

DISTRACTOR = "distractor"
MISSION = "mission"
PAIR_SUCCESS = "object pair success"
PAIR_TEST_SUCCESS = "object pair test success"
PAIR_COUNT = "object pair count"
TEST_EPISODE_SUCCESS = "test episode success"


class Args(base_main.Args, env.Args):
    attributes: str = "name"
    clip: bool = False
    freeze_keys: bool = False
    gpt_completions: bool = False
    gpt_embeddings: bool = False
    image_size: int = 256
    large_architecture: bool = False
    num_test_envs: int = 8
    num_test_names: int = 2
    pair_log_interval_coef: float = 0.01
    prefix_length: int = 0
    temp: float = None
    tgt_success_prob: float = None
    train_ln: bool = False
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        base_main.configure_logger_args(self)


class ArgsType(base_main.ArgsType, Args):
    pass


@dataclass
class Counters(base_main.Counters):
    episode_success: List[bool] = field(default_factory=list)
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    success_per_pair: DefaultDict[Tuple[str, str], List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    test_success_per_pair: DefaultDict[Tuple[str, str], List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    test_episode_success: List[bool] = field(default_factory=list)


def get_data_path_meshes(
    data_path: Path,
    obj_pattern: str,
    png_pattern: str,
):
    if data_path:
        data_path = data_path.expanduser()
        if not data_path.exists():
            raise RuntimeError(
                f"""\
        {data_path} does not exist.
        Download dataset using https://github.com/sea-bass/ycb-tools
        """
            )

        def get_names(path: Path):
            name = path.parent.parent.name
            name = re.sub(r"\d+(-[a-z])?_", "", name)
            return name.replace("_", " ")

        objs = {get_names(path): path for path in data_path.glob(obj_pattern)}
        pngs = {get_names(path): path for path in data_path.glob(png_pattern)}
        for n in objs:
            yield Mesh(objs.get(n), pngs.get(n), n)


class Trainer(base_main.Trainer):
    @classmethod
    def build_counters(cls):
        return Counters()

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def clip_processor():
        return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @classmethod
    def charts(cls, args: Args):
        kwargs = dict(visualizer_url=args.visualizer_url)
        return (
            [
                line_chart.spec(x=base_main.STEP, y=y, **kwargs)
                for y in [
                    EPISODE_SUCCESS,
                    TEST_EPISODE_SUCCESS,
                ]
            ]
            + [
                heatmap.spec(
                    x=DISTRACTOR,
                    y=MISSION,
                    color=color,
                )
                for color in [PAIR_SUCCESS, PAIR_TEST_SUCCESS]
            ]
            + super().charts(args=args)
        )

    @classmethod
    def log(
        cls,
        args: Args,
        log: dict,
        logger: HasuraLogger,
        step: int,
        counters: Counters = None,
    ):
        n_objects = 60 if args.names is None else len(args.names)
        if args.num_test_names is not None:
            n_objects -= args.num_test_names
        timesteps_per_log = args.num_processes * args.num_steps * args.log_interval
        threshold = timesteps_per_log * args.pair_log_interval_coef
        for key, success_per_pair, threshold in [
            (PAIR_SUCCESS, counters.success_per_pair, threshold),
            (PAIR_TEST_SUCCESS, counters.test_success_per_pair, 0),
        ]:
            success_per_pair = deepcopy(success_per_pair)
            for (mission, distractor), v in success_per_pair.items():
                if len(v) >= threshold:
                    super().log(
                        args=args,
                        log={
                            key: np.mean(v),
                            MISSION: mission,
                            DISTRACTOR: distractor,
                        },
                        logger=logger,
                        step=step,
                    )
                    success_per_pair[mission, distractor] = []

        if counters.episode_success:
            log.update({EPISODE_SUCCESS: np.mean(counters.episode_success)})
        if counters.test_episode_success:
            log.update({TEST_EPISODE_SUCCESS: np.mean(counters.test_episode_success)})
        super().log(args=args, log=log, logger=logger, step=step, counters=counters)
        counters.episode_success = []
        counters.fail_seed_success = []
        counters.fail_seed_usage = []
        counters.num_fail_seeds = []
        counters.test_episode_success = []

    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=action_space,
            clip=args.clip,
            gpt_embeddings=args.gpt_embeddings,
            hidden_size=args.hidden_size,
            image_size=args.image_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
            large_architecture=args.large_architecture,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
        )

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            clip_processor: CLIPProcessor,
            seed: int,
            wrap_mission_preprocessor: Callable[[gym.Env], MissionPreprocessor],
            **_kwargs,
        ):
            _env = Env(
                **{
                    k: v
                    for k, v in _kwargs.items()
                    if k in signature(Env.__init__).parameters
                },
            )
            _env.seed(seed)
            if render:
                _env = RenderWrapper(_env, mode="ascii")

            _env = ObsWrapper(_env)
            _env = ActionSpaceWrapper(_env)
            _env = CLIPProcessorWrapper(_env, clip_processor)
            _env = wrap_mission_preprocessor(_env)
            _env = RolloutsWrapper(_env)
            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    # noinspection PyMethodOverriding
    @classmethod
    def make_vec_envs(
        cls,
        attributes: str,
        clip: bool,
        data_path: str,
        gpt_completions: bool,
        gpt_embeddings: bool,
        names: Optional[str],
        num_processes: int,
        num_test_envs: int,
        num_test_names: int,
        obj_pattern: str,
        png_pattern: str,
        render: bool,
        seed: int,
        sync_envs: bool,
        test: bool,
        **kwargs,
    ):
        if test:
            num_processes = cls._num_eval_processes(num_processes, num_test_envs)

        # df = pd.read_csv("habitat.csv")
        kwargs.update(config=habitat.get_config("objectnav_mp3d.yaml"))
        tokenizer = cls.tokenizer()

        def wrap_mission_preprocessor(env: Env):
            objects = [(o,) for o in env.objects]
            if gpt_embeddings:
                return GPT3Wrapper(env, all_missions=objects)
            else:
                return TokenizerWrapper(env, all_missions=objects, tokenizer=tokenizer)

        clip_processor = cls.clip_processor() if clip else None
        return super().make_vec_envs(
            all_missions=[],  # all_missions,
            clip_processor=clip_processor,
            features={},  # features,
            num_processes=num_processes,
            render=render,
            seed=seed,
            sync_envs=sync_envs,
            test=test,
            wrap_mission_preprocessor=wrap_mission_preprocessor,
            **kwargs,
        )

    @staticmethod
    def _num_eval_processes(num_processes: int, num_test_envs):
        return min(num_processes, num_test_envs)

    @classmethod
    def num_eval_processes(cls, args: Args):
        return cls._num_eval_processes(args.num_processes, args.num_test_envs)

    @classmethod
    def process_info(cls, counters: Counters, info: dict, test: bool):
        super().process_info(counters, info, test)
        if EPISODE_SUCCESS in info:
            if test:
                counters.test_episode_success.append(info[EPISODE_SUCCESS])
                counters.test_success_per_pair[info[PAIR]].append(info[EPISODE_SUCCESS])
            else:
                counters.episode_success.append(info[EPISODE_SUCCESS])
                counters.success_per_pair[info[PAIR]].append(info[EPISODE_SUCCESS])
                counters.pairs.append(info[PAIR])

    @staticmethod
    def recurrent(args: Args):
        assert args.recurrent
        return args.recurrent

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def tokenizer():
        return GPT2Tokenizer.from_pretrained("gpt2")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def train_test_split(
        names: Tuple[str], num_test: int, rng: np.random.Generator
    ) -> TrainTest[Set[str]]:
        test_set = (
            set(rng.choice(list(names), size=num_test, replace=False))
            if num_test
            else set()
        )
        train_set = set(names) - test_set
        return TrainTest(train=train_set, test=test_set)

    @classmethod
    def update_arg(cls, args: Args, bad_attrs: List[str], k: str, v) -> None:
        if k == "num_processes" and v != base_main.DEFAULT_NUM_PROCESSES:
            return
        super().update_arg(args, bad_attrs, k, v)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
