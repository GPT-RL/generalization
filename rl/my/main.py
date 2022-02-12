import functools
from inspect import signature
from typing import Literal, Set, cast

import base_main
import numpy as np
from envs import RenderWrapper, VecPyTorch
from my.agent import Agent
from my.env import (
    OBJECTS,
    ActionInObsWrapper,
    FullyObsWrapper,
    OmitActionWrapper,
    PickupEnv,
    RolloutsWrapper,
    SuccessWrapper,
    TokenizerWrapper,
)
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer


class Args(base_main.Args):
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
    room_size: int = 5
    second_layer: bool = False
    strict: bool = True

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        base_main.configure_logger_args(self)


class ArgsType(base_main.ArgsType, Args):
    pass


class Trainer(base_main.Trainer):
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
            test: bool,
            test_objects: Set[str],
            tokenizer: GPT2Tokenizer,
            **_kwargs,
        ):

            objects = test_objects if test else set(OBJECTS) - test_objects
            objects = [o.split() for o in objects]
            objects = [(t, c) for (c, t) in objects]
            if test:
                (c1, t1), (c2, t2) = objects
                cross_product = [(c1, t2), (c2, t1)]
                objects.extend(cross_product)
                _kwargs.update(prohibited=set(cross_product))

            _env = PickupEnv(
                objects=objects,
                **{
                    k: v
                    for k, v in _kwargs.items()
                    if k in signature(PickupEnv.__init__).parameters
                },
            )
            _env = OmitActionWrapper(_env)
            #     "a really long string that I just added for testing purposes. a really long string that I just "
            #     "added for testing purposes. "
            # ]
            longest_mission: str = max(OBJECTS, key=len)

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

        return functools.partial(_thunk, **kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def tokenizer(pretrained_model):
        return GPT2Tokenizer.from_pretrained(pretrained_model)

    @classmethod
    def make_vec_envs(cls, *args, pretrained_model: str, seed: int, **kwargs):
        rng = np.random.default_rng(seed=seed)
        colors, _ = zip(*[o.split() for o in OBJECTS])
        test_color1, test_color2 = rng.choice(list(colors), size=2, replace=False)
        test_objects = {f"{test_color1} box", f"{test_color2} ball"}

        return super().make_vec_envs(
            *args,
            **kwargs,
            seed=seed,
            test_objects=test_objects,
            tokenizer=cls.tokenizer(pretrained_model),
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
