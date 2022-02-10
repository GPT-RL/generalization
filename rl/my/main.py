import functools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, cast

import base_main
import heatmap
import line_chart
import numpy as np
import pandas as pd
from base_main import RUN_ID, STEP
from envs import RenderWrapper, VecPyTorch
from my import env
from my.agent import Agent
from my.env import DESCRIPTION, EXCLUDED, Env, Mesh
from my.mesh_paths import get_meshes
from run_logger import HasuraLogger
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer
from wrappers import (
    FeatureWrapper,
    ImageNormalizerWrapper,
    RolloutsWrapper,
    SuccessWrapper,
    TokenizerWrapper,
    TrainTest,
)

EPISODE_SUCCESS = "episode success"
DISTRACTOR = "distractor"
MISSION = "mission"
PAIR = "object pair"
PAIR_SUCCESS = "object pair success"
TEST_EPISODE_SUCCESS = "test episode success"


class Args(base_main.Args, env.Args):
    num_test_envs: int = 8
    num_test_names: int = 2
    pretrained_model: Literal[
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "bert-base-uncased",
        "bert-large-uncased",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
    ] = "gpt2-large"  # what size of pretrained GPT to use
    prefix_length: int = 0
    use_features: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        base_main.configure_logger_args(self)


class ArgsType(base_main.ArgsType, Args):
    pass


@dataclass
class Counters(base_main.Counters):
    episode_success: List[bool] = field(default_factory=list)
    object_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def reset(self):
        super().reset()
        self.episode_success = []
        self.object_pairs = []


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

    @classmethod
    def charts(cls, args: Args):
        kwargs = dict(visualizer_url=args.visualizer_url)
        return [
            line_chart.spec(x=base_main.STEP, y=EPISODE_SUCCESS, **kwargs),
            line_chart.spec(x=base_main.STEP, y=TEST_EPISODE_SUCCESS, **kwargs),
            heatmap.spec(
                x=DISTRACTOR,
                y=MISSION,
                color=PAIR_SUCCESS,
                history_len=args.num_processes
                * args.num_steps
                * args.log_interval
                * 10,
            ),
            *super().charts(args=args),
        ]

    @classmethod
    def log(
        cls,
        logger: HasuraLogger,
        log: dict,
        counters: Counters = None,
        total_num_steps: int = None,
    ):
        breakpoint()
        success_per_pair = defaultdict(list)
        assert len(counters.episode_success) == len(counters.object_pairs)
        for success, pairs in zip(counters.episode_success, counters.object_pairs):
            success_per_pair[pairs].append(success)

        for (mission, distractor), v in success_per_pair.items():
            _log = {
                PAIR_SUCCESS: np.mean(v),
                MISSION: mission,
                DISTRACTOR: distractor,
                STEP: total_num_steps,
            }
            if logger.run_id is not None:
                _log.update({RUN_ID: logger.run_id})
            # super().log(logger=logger, log=_log, total_num_steps=total_num_steps)

        log.update({EPISODE_SUCCESS: np.mean(counters.episode_success)})
        super().log(
            logger=logger, log=log, counters=counters, total_num_steps=total_num_steps
        )

    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=action_space,
            pretrained_model=args.pretrained_model,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
        )

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            all_missions: list,
            features: Dict[str, List[str]],
            meshes: List[Mesh],
            rank: int,
            room_size: int,
            seed: int,
            tokenizer: GPT2Tokenizer,
            **_,
        ):

            _env = Env(meshes=meshes, rank=rank, seed=seed, size=room_size)
            _env = SuccessWrapper(_env)
            if render:
                _env = RenderWrapper(_env, mode="ascii")
            _env = ImageNormalizerWrapper(_env)
            if features:
                _env = FeatureWrapper(_env, features)
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
        data_path: str,
        pretrained_model: str,
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
        use_features: bool,
        **kwargs,
    ):
        if test:
            num_processes = cls._num_eval_processes(num_processes, num_test_envs)

        meshes = get_meshes(data_path=Path(data_path).expanduser(), names=names)

        if use_features:
            df = pd.read_csv("ycb.csv", index_col="name")
            df = df[[DESCRIPTION, EXCLUDED]]
            df = df[~df[EXCLUDED]].drop(EXCLUDED, axis=1)
            features = df.apply(
                func=lambda r: []
                if r[DESCRIPTION].isna()
                else r[DESCRIPTION].split(","),
                axis=1,
            )
            features = features.to_dict()
            meshes: List[Mesh] = [m for m in meshes if m.name in features]
            all_missions = list(features.values())
        else:
            features = None
            all_missions = [m.name for m in meshes]

        rng = np.random.default_rng(seed=seed)
        names: TrainTest[Set[str]] = cls.train_test_split(
            tuple([m.name for m in meshes]), num_test_names, rng
        )
        names: Set[str] = names.test if test else names.train
        assert len(names) > 1
        meshes = [m for m in meshes if m.name in names]

        return super().make_vec_envs(
            all_missions=all_missions,
            features=features,
            meshes=meshes,
            num_processes=num_processes,
            render=render,
            seed=seed,
            sync_envs=sync_envs,
            tokenizer=cls.tokenizer(pretrained_model),
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
    def process_info(cls, counters: Counters, info: dict):
        super().process_info(counters, info)
        if "success" in info:
            counters.episode_success.append(info["success"])
        if "pair" in info:
            counters.object_pairs.append(info["pair"])

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def tokenizer(pretrained_model):
        return GPT2Tokenizer.from_pretrained(pretrained_model)

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
