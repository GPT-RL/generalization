import functools
from typing import List, Literal, cast

import gym
import main
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    FullyObsWrapper,
    NormalizeColorsWrapper,
    PickupEnv,
    PlantAnimalWrapper,
    RenderColorPickupEnv,
    RGBImgObsWithDirectionWrapper,
    RGBtoRYBWrapper,
    RolloutsWrapper,
    TokenizerWrapper,
    ZeroOneRewardWrapper,
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
        # if "gpt" in args.pretrained_model:
        #     tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)
        # elif "bert" in args.pretrained_model:
        #     tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        # else:
        #     raise RuntimeError(f"Invalid model name: {args.pretrained_model}")
        # encoded = [tokenizer.encode(m, return_tensors="pt") for m in missions]
        # encoded = [torch.squeeze(m, 0) for m in encoded]
        # pad_token_id = tokenizer.pad_token_id
        # if pad_token_id is None:
        #     pad_token_id = tokenizer.eos_token_id
        # encoded = pad_sequence(encoded, padding_value=pad_token_id).T
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
            pretrained_model=args.pretrained_model,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
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
            num_dists: int,
            room_size: int,
            seed: int,
            strict: bool,
            test: bool,
            test_colors: str,
            tokenizer: GPT2Tokenizer,
            train_colors: str,
            **_,
        ):
            _kwargs = dict(room_size=room_size, strict=strict, seed=seed)
            if env_id == "plant-animal":
                objects = {*PlantAnimalWrapper.replacements.keys()}
                test_objects = {
                    PlantAnimalWrapper.purple_animal,
                    PlantAnimalWrapper.black_plant,
                }
                objects = test_objects if test else objects - test_objects
                objects = [o.split() for o in objects]
                objects = [(t, c) for (c, t) in objects]
                kwargs.update(room_objects=objects)
                _env = PickupEnv(objects=objects, **_kwargs)
                _env = PlantAnimalWrapper(_env)
                longest_mission = "pick up the grasshopper"

                def missions():
                    for _, vs in _env.replacements.items():
                        for v in vs:
                            yield f"pick up the {v}"

            elif env_id == "colors":
                test_colors = test_colors.split(",")
                train_colors = train_colors.split(",")
                ball = "ball"
                train_objects = sorted({(ball, col) for col in train_colors})
                test_objects = sorted({(ball, col) for col in test_colors})
                objects = test_objects if test else train_objects
                _env = RenderColorPickupEnv(
                    *args, num_dists=num_dists, objects=objects, **_kwargs
                )

                def missions():
                    for obj, col in train_objects + test_objects:
                        yield f"pick up the {col} {obj}"

            else:
                raise RuntimeError(f"{env_id} is not a valid env_id")

            missions = list(missions())
            _env = FullyObsWrapper(_env)
            if env_id == "colors":
                _env = RGBImgObsWithDirectionWrapper(_env)
                _env = RGBtoRYBWrapper(_env)
                _env = NormalizeColorsWrapper(_env)

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
    def make_vec_envs(cls, *args, pretrained_model: str, **kwargs):
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        return super().make_vec_envs(*args, **kwargs, tokenizer=tokenizer)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
