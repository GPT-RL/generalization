import functools
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import DefaultDict, List, Optional, Set, Tuple, cast

import base_main
import gym
import habitat
import heatmap
import line_chart
import numpy as np
import torch
from envs import VecPyTorch
from my import env
from my.agent import Agent
from my.env import PAIR, Env, Mesh, Obs
from run_logger import HasuraLogger
from stable_baselines3.common.monitor import Monitor
from transformers import CLIPProcessor, GPT2Tokenizer
from wrappers import (
    EPISODE_SUCCESS,
    CLIPProcessorWrapper,
    GPT3Tokenizer,
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
    large_architecture: bool = False
    num_test_envs: int = 8
    num_test_names: int = 2
    pair_log_interval_coef: float = 0.01
    prefix_length: int = 0
    qkv: bool = False
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
        features = None
        if args.qkv:
            tokenizer = cls.tokenizer(args.gpt_embeddings)
            features, *_ = envs.get_attr("features")
            features = {(word,) for words in features for word in words}
            features = list(features)
            _, d = tuple(Obs(**observation_space.spaces).mission.nvec.shape)
            tokens = [
                TokenizerWrapper.new_mission(
                    tokenizer=tokenizer, mission=features, mission_shape=(1, d)
                )
                for features in features
            ]
            features = torch.tensor(np.concatenate(tokens, axis=0))
        return cls._make_agent(
            action_space=action_space,
            args=args,
            features=features,
            observation_space=observation_space,
        )

    @classmethod
    def _make_agent(
        cls,
        action_space: gym.Space,
        args: Args,
        features: torch.Tensor,
        observation_space: gym.Space,
        agent_class: type = Agent,
        **kwargs,
    ):
        return agent_class(
            action_space=action_space,
            clip=args.clip,
            device=cls.device(cls.cuda(args)),
            freeze_keys=args.freeze_keys,
            features=features,
            gpt_embeddings=args.gpt_embeddings,
            hidden_size=args.hidden_size,
            mission_size=GPT3Tokenizer().n_embed,
            observation_space=observation_space,
            pad_token_id=cls.tokenizer(args.gpt_embeddings).eos_token_id,
            qkv=args.qkv,
            recurrent=cls.recurrent(args),
            large_architecture=args.large_architecture,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
            **kwargs,
        )

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            all_missions: list,
            clip_processor: CLIPProcessor,
            seed: int,
            test: bool,
            tokenizer: GPT2Tokenizer,
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

            _env = CLIPProcessorWrapper(_env, clip_processor)
            _env = TokenizerWrapper(
                _env,
                tokenizer=tokenizer,
                all_missions=all_missions,
            )
            _env = RolloutsWrapper(_env)
            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    # noinspection PyMethodOverriding
    @classmethod
    def make_vec_envs(
        cls,
        attributes: str,
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

        return super().make_vec_envs(
            all_missions=[],  # all_missions,
            clip_processor=cls.clip_processor(),
            features={},  # features,
            num_processes=num_processes,
            render=render,
            seed=seed,
            sync_envs=sync_envs,
            tokenizer=cls.tokenizer(gpt_embeddings),
            test=test,
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
    def tokenizer(gpt_embeddings: bool):
        if gpt_embeddings:
            return GPT3Tokenizer()
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


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
