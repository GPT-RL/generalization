import functools
import itertools
import pickle
from pathlib import Path
from pprint import pprint
from typing import List, Literal, NamedTuple, Set, cast

import gym
import main
import numpy as np
from babyai_agent import Agent
from babyai_env import (
    ActionInObsWrapper,
    BabyAIEnv,
    FullyObsWrapper,
    MultiSeedWrapper,
    RolloutsWrapper,
    SuccessWrapper,
    TokenizerWrapper,
    TrainTestChecker,
)
from colors import color as ansi_color
from envs import RenderWrapper, VecPyTorch
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from transformers import GPT2Tokenizer


class MissionsSeeds(NamedTuple):
    missions: Set[str]
    seeds: Set[int]


class TrainTestSplits(NamedTuple):
    train: MissionsSeeds
    test: MissionsSeeds


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
    envs_per_mission: int = 2
    instr_kinds: str = "action"
    missions_per_split: int = 2
    locations: bool = False
    locked_room_prob: float = 0
    num_dists: int = 1
    num_rows: int = 1
    room_size: int = 5
    second_layer: bool = False
    strict: bool = True
    instr_path: str = "/root/.cache/data/instructions.pkl"
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
        instr_kinds = set(args.instr_kinds.split(","))
        if instr_kinds & {"and", "seq"}:
            assert args.recurrent
        return args.recurrent

    @classmethod
    def make_env(cls, **kwargs):
        def _thunk(
            allow_early_resets: bool,
            test: bool,
            train_test_splits: TrainTestSplits,
            tokenizer: GPT2Tokenizer,
            render: bool = False,
            **_kwargs,
        ):
            _env = BabyAIEnv(**_kwargs)
            _env = MultiSeedWrapper(
                _env,
                seeds=list(
                    train_test_splits.train.seeds
                    if test
                    else train_test_splits.test.seeds
                ),
            )
            _env = TrainTestChecker(
                _env,
                missions=train_test_splits.train.missions
                if test
                else train_test_splits.test.missions,
            )
            longest_mission = (
                "put a ball next to a purple door after you put a blue box next to a grey box and pick "
                "up the purple box "
            )

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

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def train_test_splits(
        envs_per_mission: int,
        missions_per_split: int,
        seed: int,
        **kwargs,
    ):
        env = BabyAIEnv(**kwargs, seed=seed)
        total_missions = 2 * missions_per_split
        missions_to_seeds = {}
        rng = np.random.default_rng(seed=seed)

        with tqdm(
            desc="Generating mission/environment pairs...",
            total=envs_per_mission * missions_per_split * 2,
        ) as bar:
            for i in itertools.count():
                if i % 10_000 == 0:
                    empty_missions = {
                        k: v for k, v in missions_to_seeds.items() if len(v) == 0
                    }
                    if empty_missions:
                        print(
                            ansi_color(
                                "missions with no environments", style="underline"
                            )
                        )
                    for k in empty_missions:
                        print(k)
                        del missions_to_seeds[k]

                    while len(missions_to_seeds) < total_missions:
                        env.seed(int(rng.choice(1_000_000)))
                        instr = env.rand_instr(env.action_kinds, env.instr_kinds)
                        mission = instr.surface(env)
                        if mission not in missions_to_seeds:
                            missions_to_seeds[mission] = []

                num_seeds = [len(v) for v in missions_to_seeds.values()]
                if all([n >= envs_per_mission for n in num_seeds]):
                    break

                seed_i = seed + i
                env.seed(seed_i)
                mission = env.reset()["mission"]
                if mission in missions_to_seeds:
                    seeds = missions_to_seeds[mission]
                    if len(seeds) < envs_per_mission:
                        bar.update(1)
                        seeds.append(seed_i)

        assert len(missions_to_seeds) == total_missions
        for v in missions_to_seeds.values():
            assert len(v) == envs_per_mission
        missions = set(missions_to_seeds)
        train_missions = set(
            rng.choice(
                list(missions_to_seeds.keys()), replace=False, size=missions_per_split
            )
        )
        test_missions = missions - train_missions

        def get_seeds(_missions: Set[str]):
            for _mission in _missions:
                for _seed in missions_to_seeds[_mission]:
                    yield _seed

        def missions_and_seeds(_missions: Set[str]):
            return MissionsSeeds(missions=_missions, seeds=set(get_seeds(_missions)))

        return TrainTestSplits(
            train=missions_and_seeds(train_missions),
            test=missions_and_seeds(test_missions),
        )

    @classmethod
    def make_vec_envs(
        cls,
        *args,
        missions_per_split: int,
        envs_per_mission: int,
        pretrained_model: str,
        test: bool,
        instr_path: str,
        **kwargs,
    ):
        train_test_splits = None
        path = Path(instr_path)
        if path.exists():
            with path.open("rb") as f:
                cached: TrainTestSplits = pickle.load(f)
            if (
                envs_per_mission * missions_per_split
                == len(cached.test.seeds)
                == len(cached.train.seeds)
            ) and (
                missions_per_split
                == len(cached.test.missions)
                == len(cached.train.missions)
            ):
                train_test_splits = cached
        if train_test_splits is None:
            train_test_splits = cls.train_test_splits(
                envs_per_mission=envs_per_mission,
                missions_per_split=missions_per_split,
                **kwargs,
            )
            with path.open("wb") as f:
                pickle.dump(train_test_splits, f)

        if test:
            print("Test missions:")
            pprint(train_test_splits.test.missions)
        else:
            print("Train missions:")
            pprint(train_test_splits.train.missions)

        return super().make_vec_envs(
            *args,
            **kwargs,
            test=test,
            train_test_splits=train_test_splits,
            tokenizer=cls.tokenizer(pretrained_model),
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
