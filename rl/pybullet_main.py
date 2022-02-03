import csv
import functools
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, cast

import main
import numpy as np
import pybullet_env
from envs import RenderWrapper, VecPyTorch
from pybullet_agent import Agent
from pybullet_env import URDF, Env, get_urdfs
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer
from wrappers import (
    FeatureWrapper,
    ImageNormalizerWrapper,
    RolloutsWrapper,
    TokenizerWrapper,
    TrainTest,
    VideoRecorderWrapper,
)

ROUNDED_STEP = "rounded step"
ENV = "environment ID"
ENV_RETURN = "environment return"


class Args(main.Args, pybullet_env.Args):
    data_path: str = "/root/.cache/data/pybullet-URDF-models/urdf_models/models"
    names: Optional[str] = None
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
    record: bool = False
    use_features: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


@dataclass
class Counters(main.Counters):
    episode_rewards_per_env: dict = field(default_factory=lambda: defaultdict(list))

    def reset(self):
        super().reset()
        self.episode_rewards_per_env = defaultdict(list)


class Trainer(main.Trainer):
    @classmethod
    def build_counters(cls):
        return Counters()

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
            env_bounds: float,
            features: Dict[str, List[str]],
            image_size: int,
            max_episode_steps: int,
            rank: int,
            record: bool,
            run_id: Optional[int],
            seed: int,
            step_size: float,
            steps_per_action: int,
            tokenizer: GPT2Tokenizer,
            urdfs: List[Tuple[URDF, URDF]],
            **_,
        ):

            _env = Env(
                env_bounds=env_bounds,
                image_size=image_size,
                max_episode_steps=max_episode_steps,
                random_seed=seed,
                rank=rank,
                step_size=step_size,
                steps_per_action=steps_per_action,
                urdfs=urdfs[rank],
            )
            if record:
                video_path = Path(cls.save_dir(run_id), "video.mp4")
                _env = VideoRecorderWrapper(_env, path=video_path)
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
        render: bool,
        seed: int,
        sync_envs: bool,
        test: bool,
        use_features: bool,
        **kwargs,
    ):
        if test:
            num_processes = cls._num_eval_processes(num_processes, num_test_envs)

        data_path = Path(data_path)
        if not data_path.exists():
            raise RuntimeError(
                f"""\
{data_path} does not exist.
Download dataset using: git clone git@github.com:GPT-RL/pybullet-URDF-models.git
"""
            )

        if names:
            names: Set[str] = set(names.split(","))
        urdfs = list(get_urdfs(data_path, names))
        names: List[str] = [urdf.name for urdf in urdfs]

        if use_features:
            with Path("features.csv").open() as f:

                def get_features():
                    for k, *vs in csv.reader(f):
                        vs = [v for v in vs if v]
                        if vs:
                            yield k, tuple(vs)

                features = dict(get_features())
            urdfs = [u for u in urdfs if u.name in features]
            all_missions = list(features.values())
        else:
            features = None
            all_missions = names

        rng = np.random.default_rng(seed=seed)
        names: TrainTest[Set[str]] = cls.train_test_split(
            tuple(names), num_test_names, rng
        )
        names: Set[str] = names.test if test else names.train
        assert len(names) > 1
        urdfs = [u for u in urdfs if u.name in names]

        def get_pairs():
            while True:
                rng.shuffle(urdfs)
                for urdf in urdfs:
                    opposites = [u for u in urdfs if u.name != urdf.name]
                    opposite = opposites[rng.choice(len(opposites))]
                    yield urdf, opposite

        pairs = list(itertools.islice(get_pairs(), num_processes))

        return super().make_vec_envs(
            all_missions=all_missions,
            features=features,
            num_processes=num_processes,
            pairs=pairs,
            render=render,
            seed=seed,
            sync_envs=sync_envs,
            tokenizer=cls.tokenizer(pretrained_model),
            test=test,
            urdfs=pairs,
            **kwargs,
        )

    @staticmethod
    def _num_eval_processes(num_processes: int, num_test_envs):
        return min(num_processes, num_test_envs)

    @classmethod
    def num_eval_processes(cls, args: Args):
        return cls._num_eval_processes(args.num_processes, args.num_test_envs)

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
