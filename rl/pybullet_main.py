import functools
from pathlib import Path
from typing import List, Literal, cast

import main
import numpy as np
from envs import RenderWrapper, VecPyTorch
from gym_minigrid.minigrid import COLORS
from pybullet_agent import Agent
from pybullet_env import URDF, Env, get_model_ids, get_urdfs
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer
from wrappers import RolloutsWrapper, alt_type


class Args(main.Args):
    data_path: str = "/root/.cache/data/dataset"
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
            seed: int,
            tokenizer: GPT2Tokenizer,
            urdfs: List[URDF],
            **_,
        ):
            _env = Env(tokenizer, urdfs)
            _env.seed(seed)

            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    # noinspection PyShadowingBuiltins
    @staticmethod
    @functools.lru_cache(maxsize=2)
    def stock_prefix(type: str, prefix_length: int, seed: int):
        if prefix_length == 0:
            return ""
        rng = np.random.default_rng(seed=seed)
        colors = rng.choice(list(COLORS), replace=False, size=prefix_length)
        return ". ".join(
            [f"{color} {alt_type(type)}: {color} {type}" for color in colors]
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def tokenizer(pretrained_model):
        return GPT2Tokenizer.from_pretrained(pretrained_model)

    @classmethod
    def make_vec_envs(
        cls,
        *args,
        data_path: str,
        prefix_length: int,
        pretrained_model: str,
        seed: int,
        **kwargs,
    ):
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

        urdfs = list(get_urdfs(data_path, get_model_ids()))

        return super().make_vec_envs(
            *args,
            **kwargs,
            seed=seed,
            urdfs=urdfs,
            tokenizer=cls.tokenizer(pretrained_model),
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
