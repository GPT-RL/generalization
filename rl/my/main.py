import functools
from inspect import signature
from typing import Literal, Set, cast

import base_main
import gym
import numpy as np
import torch
from envs import RenderWrapper, VecPyTorch
from gym.spaces import Discrete
from my.agent import Agent
from my.env import (
    OBJECTS,
    ActionInObsWrapper,
    BabyAIEnv,
    FullyObsWrapper,
    Obs,
    OmitActionWrapper,
    RolloutsWrapper,
    SuccessWrapper,
    TokenizerWrapper,
)
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer


class Args(base_main.Args):
    action_kinds: str = "pickup"
    attn_temp: float = 5
    freeze_keys: bool = False
    qkv: bool = False
    num_rows: int = 1
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
    split_words: bool = False
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
        missions = None
        cuda = cls.cuda(args)
        device = cls.device(cuda)
        if args.qkv:
            tokenizer = cls.tokenizer(args.pretrained_model)
            missions, *_ = envs.get_attr("missions")
            mission_shape = tuple(Obs(*observation_space.spaces).mission.nvec.shape)
            tokens = [
                TokenizerWrapper.new_mission(
                    tokenizer=tokenizer, mission=mission, mission_shape=mission_shape
                )
                for mission in missions
            ]
            missions = torch.Tensor(tokens).long()
        return cls._make_agent(
            action_space=action_space,
            args=args,
            device=device,
            missions=missions,
            observation_space=observation_space,
        )

    @classmethod
    def _make_agent(
        cls,
        action_space: Discrete,
        args: Args,
        device: torch.device,
        missions: list,
        observation_space: gym.spaces.Dict,
    ):
        return Agent(
            action_space=action_space,
            attn_temp=args.attn_temp,
            device=device,
            freeze_keys=args.freeze_keys,
            hidden_size=args.hidden_size,
            qkv=args.qkv,
            observation_space=observation_space,
            missions=missions,
            pretrained_model=args.pretrained_model,
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
            action_kinds: str,
            split_words: bool,
            test: bool,
            test_objects: Set[str],
            tokenizer: GPT2Tokenizer,
            **_kwargs,
        ):

            all_missions = (
                [tuple(o.split()) for o in OBJECTS] if split_words else OBJECTS
            )
            objects = test_objects if test else set(OBJECTS) - test_objects
            objects = [o.split() for o in objects]
            objects = [(t, c) for (c, t) in objects]
            if test:
                (c1, t1), (c2, t2) = objects
                cross_product = [(c1, t2), (c2, t1)]
                objects.extend(cross_product)
                _kwargs.update(prohibited=set(cross_product))

            _env = BabyAIEnv(
                action_kinds=action_kinds.split(","),
                missions=all_missions,
                objects=objects,
                **{
                    k: v
                    for k, v in _kwargs.items()
                    if k in signature(BabyAIEnv.__init__).parameters
                },
            )
            _env = OmitActionWrapper(_env, split_words)
            #     "a really long string that I just added for testing purposes. a really long string that I just "
            #     "added for testing purposes. "
            # ]

            _env = FullyObsWrapper(_env)
            _env = ActionInObsWrapper(_env)
            _env = SuccessWrapper(_env)
            _env = TokenizerWrapper(
                _env,
                tokenizer=tokenizer,
                all_missions=all_missions,
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
