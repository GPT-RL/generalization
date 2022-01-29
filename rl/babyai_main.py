import functools
from typing import List, Literal, Set, cast

import main
import numpy as np
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    FullyObsWrapper,
    PickupEnv,
    PlantAnimalWrapper,
    RolloutsWrapper,
    SuccessWrapper,
    TokenizerWrapper,
    alt_type,
)
from envs import RenderWrapper, VecPyTorch
from gym_minigrid.minigrid import COLORS
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
    split_words: bool = False
    strict: bool = True
    prefix_length: int = 4

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
            split_words=args.split_words,
        )

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            env_id: str,
            room_size: int,
            seed: int,
            prefixes: dict,
            split_words: bool,
            strict: bool,
            test: bool,
            test_objects: Set[str],
            tokenizer: GPT2Tokenizer,
            **_,
        ):
            if env_id == "plant-animal":
                objects = set(PlantAnimalWrapper.replacements)
                objects = test_objects if test else objects - test_objects
                objects = [o.split() for o in objects]
                objects = [(t, c) for (c, t) in objects]
                kwargs = {}
                if test:
                    (c1, t1), (c2, t2) = objects
                    cross_product = [(c1, t2), (c2, t1)]
                    objects.extend(cross_product)
                    kwargs.update(prohibited=set(cross_product))

                _env = PickupEnv(
                    objects=objects,
                    room_size=room_size,
                    strict=strict,
                    seed=seed,
                    **kwargs,
                )
                _env = PlantAnimalWrapper(_env, prefixes)
                missions: List[str] = _env.missions  # + [
                #     "a really long string that I just added for testing purposes. a really long string that I just "
                #     "added for testing purposes. "
                # ]
                longest_mission = max(missions, key=len)

            else:
                raise RuntimeError(f"{env_id} is not a valid env_id")

            _env = FullyObsWrapper(_env)
            _env = ActionInObsWrapper(_env)
            _env = SuccessWrapper(_env)
            _env = TokenizerWrapper(
                _env,
                tokenizer=tokenizer,
                longest_mission=longest_mission,
                split_words=split_words,
            )
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
        cls, *args, pretrained_model: str, prefix_length: int, seed: int, **kwargs
    ):
        prefixes = {
            ty: cls.stock_prefix(ty, prefix_length=prefix_length, seed=seed)
            for ty in ["ball", "box"]
        }
        rng = np.random.default_rng(seed=seed)
        colors, _ = zip(*[o.split() for o in PlantAnimalWrapper.replacements])
        test_color1, test_color2 = rng.choice(list(colors), size=2, replace=False)
        test_objects = {f"{test_color1} box", f"{test_color2} ball"}

        return super().make_vec_envs(
            *args,
            **kwargs,
            seed=seed,
            prefixes=prefixes,
            test_objects=test_objects,
            tokenizer=cls.tokenizer(pretrained_model),
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
