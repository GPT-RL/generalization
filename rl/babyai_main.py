import functools
import pickle
from pathlib import Path
from typing import List, Literal, cast

import gym
import main
import numpy as np
from babyai.levels.verifier import (
    LOC_NAMES,
    OBJ_TYPES,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    BabyAIEnv,
    FullyObsWrapper,
    RolloutsWrapper,
    SuccessWrapper,
    TokenizerWrapper,
)
from envs import RenderWrapper, VecPyTorch
from gym_minigrid.minigrid import COLOR_NAMES
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from transformers import GPT2Tokenizer


class Args(main.Args):
    action_kinds: str = "pickup"
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
    instr_kinds: str = "action"
    locations: bool = False
    locked_room_prob: float = 0
    num_dists: int = 1
    num_rows: int = 1
    num_train: int = 30
    room_size: int = 5
    second_layer: bool = False
    strict: bool = True
    train_instructions_path: str = "/root/.cache/data/train_instructions.pkl"
    unblocking: bool = False
    use_gru: bool = False

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
            use_gru=args.use_gru,
        )

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            tokenizer: GPT2Tokenizer,
            **kwargs,
        ):
            _env = BabyAIEnv(**kwargs)
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

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def tokenizer(pretrained_model):
        return GPT2Tokenizer.from_pretrained(pretrained_model)

    @classmethod
    @functools.lru_cache(maxsize=1)
    def instructions(cls, seed, num_train, env):
        def get_objs():
            colors = [None, *COLOR_NAMES]
            types = list(OBJ_TYPES)
            locations = [None, *LOC_NAMES]

            for color in colors:
                for ty in types:
                    for loc in locations:
                        yield ObjDesc(ty, color, loc)

        def get_instrs():
            for obj in get_objs():
                yield GoToInstr(obj)
                if obj.type == "door":
                    yield OpenInstr(obj)
                else:
                    yield PickupInstr(obj)
                    for obj_fixed in get_objs():
                        yield PutNextInstr(obj, obj_fixed)

        def is_valid(instr):
            env.reset()
            for _ in range(1000):
                env.reset()
                if env.instr_is_valid(instr):
                    return True
            return False

        instrs = list(get_instrs())
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(instrs)
        train_instructions = []
        with tqdm(desc="Validating train instructions...", total=num_train) as bar:
            for instr in instrs:
                if len(train_instructions) == num_train:
                    break
                if is_valid(instr):
                    train_instructions.append(instr)
                    bar.update(1)

        train_instructions = rng.choice(
            train_instructions, replace=False, size=num_train
        )
        length = len(train_instructions)
        train_instructions = set(train_instructions)
        assert length == len(train_instructions)
        return train_instructions

    @classmethod
    def make_vec_envs(
        cls,
        *args,
        num_train: int,
        pretrained_model: str,
        seed: int,
        train_instructions_path: str,
        **kwargs,
    ):
        train_instructions = None
        path = Path(train_instructions_path)
        if path.exists():
            with path.open("rb") as f:
                cached_num_train, cached_train_instructions = pickle.load(f)
                if cached_num_train == num_train:
                    train_instructions = cached_train_instructions

        if train_instructions is None:
            _kwargs = dict(**kwargs)
            _kwargs.update(test=True)
            env = BabyAIEnv(
                seed=seed,
                train_instructions=set(),
                **_kwargs,
            )
            train_instructions = cls.instructions(seed, num_train, env)
            with path.open("wb") as f:
                pickle.dump((num_train, train_instructions), f)
        return super().make_vec_envs(
            *args,
            **kwargs,
            seed=seed,
            train_instructions=train_instructions,
            tokenizer=cls.tokenizer(pretrained_model),
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
