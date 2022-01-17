import functools
from typing import List, Literal, cast

import gym
import main
from babyai_agent import Agent
from babyai_env import (
    OBJECTS,
    ActionInObsWrapper,
    FullyObsWrapper,
    PickupEnv,
    RolloutsWrapper,
    SuccessWrapper,
    TokenizerWrapper,
)
from envs import RenderWrapper, VecPyTorch
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer


class Args(main.Args):
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
    env: str = "plant-animal"  # env ID for gym
    room_size: int = 5
    second_layer: bool = False
    strict: bool = True
    test_organisms: str = None

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
        missions: List[str]
        return cls._make_agent(
            action_space=action_space,
            observation_space=observation_space,
            args=args,
        )

    @classmethod
    def _make_agent(
        cls,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        args: ArgsType,
    ):
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
            room_size: int,
            seed: int,
            strict: bool,
            test: bool,
            test_organisms: str,
            tokenizer: GPT2Tokenizer,
            **_,
        ):
            test_objects = set(test_organisms.split(","))
            objects = test_objects if test else OBJECTS - test_objects
            objects = [o.split() for o in objects]
            objects = [(t, c) for (c, t) in objects]
            _env = PickupEnv(
                objects=objects,
                room_size=room_size,
                strict=strict,
                seed=seed,
            )
            longest_mission = "pick up the grasshopper"

            _env = FullyObsWrapper(_env)
            _env = ActionInObsWrapper(_env)
            _env = SuccessWrapper(_env)
            _env = TokenizerWrapper(
                _env,
                tokenizer=tokenizer,
                longest_mission=longest_mission,
            )
            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    @classmethod
    def make_vec_envs(cls, *args, pretrained_model: str, **kwargs):
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        return super().make_vec_envs(*args, **kwargs, tokenizer=tokenizer)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
