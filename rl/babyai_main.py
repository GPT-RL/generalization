import functools
from typing import List, Literal, cast

import gym
import torch
from stable_baselines3.common.monitor import Monitor
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

import main
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    DirectionWrapper,
    FullyObsWrapper,
    MissionEnumeratorWrapper,
    PickupEnv,
    PlantAnimalWrapper,
    RGBImgObsWithDirectionWrapper,
    RenderColorPickupEnv,
    RolloutsWrapper,
    ZeroOneRewardWrapper,
)
from envs import RenderWrapper, VecPyTorch
from utils import get_gpt_size


class Args(main.Args):
    embedding_size: Literal[
        "small", "medium", "large", "xl"
    ] = "medium"  # what size of pretrained GPT to use
    env: str = "GoToLocal"  # env ID for gym
    go_and_face_synonyms: str = None
    negation_types: str = None
    negation_colors: str = None
    num_dists: int = 1
    room_size: int = 5
    scaled_reward: bool = False
    second_layer: bool = False
    strict: bool = True
    test_colors: str = None
    train_colors: str = None
    test_descriptors: str = None
    test_number: int = None
    test_wordings: str = None
    test_walls: str = "south,southeast"
    train_wordings: str = None

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
        missions, *_ = envs.get_attr("missions")
        tokenizer = GPT2Tokenizer.from_pretrained(get_gpt_size(args.embedding_size))
        encoded = [tokenizer.encode(m, return_tensors="pt") for m in missions]
        encoded = [torch.squeeze(m, 0) for m in encoded]
        encoded = pad_sequence(encoded, padding_value=tokenizer.eos_token_id).T
        return cls._make_agent(
            action_space=action_space,
            observation_space=observation_space,
            encoded=encoded,
            args=args,
        )

    @classmethod
    def _make_agent(
        cls,
        encoded: torch.Tensor,
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
            encoded=encoded,
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
            num_dists: int,
            room_size: int,
            scaled_reward: bool,
            seed: int,
            strict: bool,
            test: bool,
            test_colors: str,
            train_colors: str,
            **_,
        ):
            _kwargs = dict(
                room_size=room_size, strict=strict, num_dists=num_dists, seed=seed
            )
            missions = None
            if env_id == "plant-animal":
                objects = {*PlantAnimalWrapper.replacements.keys()}
                test_objects = {
                    PlantAnimalWrapper.purple_animal,
                    PlantAnimalWrapper.black_plant,
                }
                objects = sorted(test_objects if test else objects - test_objects)
                objects = [o.split() for o in objects]
                objects = [(t, c) for (c, t) in objects]
                _env = PickupEnv(
                    *args, room_objects=objects, goal_objects=objects, **_kwargs
                )
                _env = PlantAnimalWrapper(_env)

                def missions():
                    for _, vs in _env.replacements.items():
                        for v in vs:
                            yield f"pick up the {v}"

                missions = list(missions())
            elif env_id == "colors":
                test_colors = test_colors.split(",")
                train_colors = train_colors.split(",")
                ball = "ball"
                train_objects = sorted({(ball, col) for col in train_colors})
                test_objects = sorted({(ball, col) for col in test_colors})
                objects = test_objects if test else train_objects
                _env = RenderColorPickupEnv(
                    *args,
                    room_objects=objects,
                    goal_objects=objects,
                    **_kwargs,
                )
            else:
                raise RuntimeError(f"{env_id} is not a valid env_id")

            _env = DirectionWrapper(_env)
            if env_id == "colors":
                _env = RGBImgObsWithDirectionWrapper(_env)
            elif env_id != "linear":
                _env = FullyObsWrapper(_env)

            _env = ActionInObsWrapper(_env)
            if not (env_id in ("go-to-loc", "go-to-row", "linear") and scaled_reward):
                _env = ZeroOneRewardWrapper(_env)
            _env = MissionEnumeratorWrapper(_env, missions=missions)
            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
