import functools
from typing import List, Literal, cast

import gym
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    FullyObsWrapper,
    PickupEnv,
    PlantAnimalWrapper,
    RolloutsWrapper,
    TokenizerWrapper,
    ZeroOneRewardWrapper,
)
from envs import RenderWrapper, VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "plant-animal"  # env ID for gym
    num_dists: int = 1
    room_size: int = 5
    second_layer: bool = False
    strict: bool = True
    test_colors: str = None
    train_colors: str = None

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
        # missions, *_ = envs.get_attr("missions")
        # tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))
        # encoded = [tokenizer.encode(m, return_tensors="pt") for m in missions]
        # encoded = [torch.squeeze(m, 0) for m in encoded]
        # encoded = pad_sequence(encoded, padding_value=tokenizer.eos_token_id).T
        return cls._make_agent(
            action_space=action_space,
            observation_space=observation_space,
            # encoded=encoded,
            args=args,
        )

    @classmethod
    def _make_agent(
        cls,
        # encoded: torch.Tensor,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        args: ArgsType,
    ):
        return Agent(
            action_space=action_space,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
            second_layer=args.second_layer,
            # encoded=encoded,
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
            strict: bool,
            test: bool,
            tokenizer: GPT2Tokenizer,
            **_,
        ):
            missions = None
            if env == "plant-animal":
                objects = {*PlantAnimalWrapper.replacements.keys()}
                test_objects = {
                    PlantAnimalWrapper.purple_animal,
                    PlantAnimalWrapper.black_plant,
                }
                room_objects = test_objects if test else objects - test_objects
                room_objects = [o.split() for o in room_objects]
                room_objects = [(t, c) for (c, t) in room_objects]
                kwargs.update(room_objects=room_objects)
                _env = PickupEnv(
                    seed=seed,
                    room_objects=room_objects,
                    goal_objects=room_objects,
                    room_size=room_size,
                    strict=strict,
                )
                _env = PlantAnimalWrapper(_env)
                longest_mission = "pick up the grasshopper"

                def missions():
                    for _, vs in _env.replacements.items():
                        for v in vs:
                            yield f"pick up the {v}"

                missions = list(missions())

            else:
                raise RuntimeError(f"{env_id} is not a valid env_id")

            _env = FullyObsWrapper(_env)
            _env = ActionInObsWrapper(_env)
            _env = ZeroOneRewardWrapper(_env)
            _env = TokenizerWrapper(
                _env,
                tokenizer=tokenizer,
                longest_mission=longest_mission,
            )
            # _env = MissionEnumeratorWrapper(_env, missions=missions)
            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    @classmethod
    def make_vec_envs(cls, *args, embedding_size: int, **kwargs):
        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(embedding_size))
        return super().make_vec_envs(*args, **kwargs, tokenizer=tokenizer)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
