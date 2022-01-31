import functools
import itertools
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple, cast

import main
import numpy as np
from envs import RenderWrapper, VecPyTorch
from pybullet_agent import Agent
from pybullet_env import URDF, Env, get_urdfs
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer
from vec_env import DummyVecEnv, SubprocVecEnv
from wrappers import RolloutsWrapper, TokenizerWrapper, TrainTest


class Args(main.Args):
    data_path: str = "/root/.cache/data/dataset"
    image_size: float = 64
    names: Optional[str] = None
    max_episode_steps: int = 200
    models: Optional[str] = None
    num_envs: int = 8
    num_test: int = 2
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
    steps_per_action: int = 5

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


class Trainer(main.Trainer):
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

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            image_size: float,
            longest_mission: str,
            max_episode_steps: int,
            seed: int,
            steps_per_action: int,
            tokenizer: GPT2Tokenizer,
            urdfs: Tuple[URDF, URDF],
            **_,
        ):
            _env = Env(
                image_size=image_size,
                is_render=render,
                max_episode_steps=max_episode_steps,
                random_seed=seed,
                steps_per_action=steps_per_action,
                urdfs=urdfs,
            )
            _env = TokenizerWrapper(
                _env, tokenizer=tokenizer, longest_mission=longest_mission
            )

            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def tokenizer(pretrained_model):
        return GPT2Tokenizer.from_pretrained(pretrained_model)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def train_test_split(
        names: Tuple[str], num_test: int, rng: np.random.Generator
    ) -> TrainTest[Set[str]]:
        test_set = set(rng.choice(list(names), size=num_test, replace=False))
        train_set = set(names) - test_set
        return TrainTest(train=train_set, test=test_set)

    # noinspection PyMethodOverriding
    @classmethod
    def _make_vec_envs(
        cls,
        data_path: str,
        num_envs: int,
        pretrained_model: str,
        models: Optional[str],
        names: Optional[str],
        num_processes: int,
        num_test: int,
        render: bool,
        seed: int,
        sync_envs: bool,
        test: bool,
        **kwargs,
    ):

        if render:
            num_envs = 1

        data_path = Path(data_path)
        if not data_path.exists():
            raise RuntimeError(
                f"""\
{data_path} does not exist.
Download dataset using:
wget 'https://sapien.ucsd.edu/api/download/partnet-mobility-v0.zip\
?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImV0aGFuYn\
JvQHVtaWNoLmVkdSIsImlwIjoiMTcyLjIwLjAuMSIsInByaXZpbGVnZSI6MSwiaWF0\
IjoxNjQzNDkyNzc3LCJleHAiOjE2NDM1NzkxNzd9.R3y0kIb11_85VHBdVgU0xRP15\
zM_ZGMrpH3vL4ECpsw'
and unzip downloaded file\
"""
            )

        # mapping = {}
        # for subdir in data_path.iterdir():
        #     with Path(subdir, "meta.json").open() as f:
        #         meta = json.load(f)
        #     name = meta["model_cat"]
        #     mapping[subdir.name] = name

        if names:
            names: Set[str] = set(names.split(","))
        if models:
            models: Set[str] = set(models.split(","))
        urdfs = list(get_urdfs(data_path, models, names))
        names: List[str] = [urdf.name for urdf in urdfs]
        longest_mission = max(names, key=len)
        rng = np.random.default_rng(seed=seed)
        names: TrainTest[Set[str]] = cls.train_test_split(tuple(names), num_test, rng)
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

        pairs = list(itertools.islice(get_pairs(), num_envs))

        envs = [
            cls.make_env(
                longest_mission=longest_mission,
                render=render,
                seed=seed + i,
                test=test,
                tokenizer=cls.tokenizer(pretrained_model=pretrained_model),
                urdfs=pairs[i],
                **kwargs,
            )
            for i in range(num_envs)
        ]

        if len(envs) == 1 or sync_envs or render:
            envs = DummyVecEnv(envs, num_processes)
        else:
            envs = SubprocVecEnv(envs, num_processes)
        return envs


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
