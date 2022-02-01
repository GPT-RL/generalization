import functools
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple, cast

import main
import numpy as np
from area_chart import spec
from envs import RenderWrapper, VecPyTorch
from pybullet_agent import Agent
from pybullet_env import URDF, Env, get_urdfs
from stable_baselines3.common.monitor import Monitor
from transformers import GPT2Tokenizer
from vec_env import DummyVecEnv, SubprocVecEnv
from wrappers import (
    ImageNormalizerWrapper,
    RolloutsWrapper,
    TokenizerWrapper,
    TrainTest,
)

ROUNDED_STEP = "rounded step"
ENV = "environment ID"
ENV_RETURN = "environment return"


class Args(main.Args):
    data_path: str = "/root/.cache/data/dataset"
    image_size: float = 84
    names: Optional[str] = None
    max_episode_steps: int = 200
    models: Optional[str] = None
    num_envs: int = 8
    num_test_envs: int = 8
    num_test_names: int = 2
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
    steps_per_action: int = 5

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


@dataclass
class Counters(main.Counters):
    episode_rewards_per_env: dict = field(default_factory=lambda: defaultdict(list))

    def reset(self):
        super().reset()
        self.episode_rewards_per_env = defaultdict(list)


class Trainer(main.Trainer):
    @classmethod
    def process_info(cls, counters: Counters, info: dict):
        super().process_info(counters, info)
        if "episode" in info.keys():
            env = info["env"]
            episode_return = info["episode"]["r"]
            counters.episode_rewards_per_env[env].append(episode_return)

    @classmethod
    def log(
        cls,
        logger,
        log: dict,
        counters: Counters,
        total_num_steps: int,
    ):
        super().log(logger, log, counters, total_num_steps)
        rounded_step = round(total_num_steps / 100) * 100
        if logger.run_id is not None:
            for k, v in counters.episode_rewards_per_env.items():
                logger.log({ROUNDED_STEP: rounded_step, ENV_RETURN: np.mean(v)})
                for _ in v:
                    logger.log({ROUNDED_STEP: rounded_step, ENV: k})

    @classmethod
    def build_counters(cls):
        return Counters()

    @classmethod
    def charts(cls, **kwargs):
        return super().charts(**kwargs) + [
            spec(x=ROUNDED_STEP, color=ENV),
            spec(x=ROUNDED_STEP, color=ENV_RETURN),
        ]

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
            image_size: float,
            longest_mission: str,
            max_episode_steps: int,
            rank: int,
            seed: int,
            steps_per_action: int,
            tokenizer: GPT2Tokenizer,
            urdfs: Tuple[URDF, URDF],
            **_,
        ):
            _env = Env(
                image_size=image_size,
                is_render=render,
                max_episode_steps=max_episode_steps,
                random_seed=seed,
                rank=rank,
                steps_per_action=steps_per_action,
                urdfs=urdfs,
            )
            _env = ImageNormalizerWrapper(_env)
            _env = TokenizerWrapper(
                _env, tokenizer=tokenizer, longest_mission=longest_mission
            )

            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, env_id=env, **kwargs)

    @staticmethod
    def _num_eval_processes(num_processes: int, num_test_envs):
        return min(num_processes, num_test_envs)

    @classmethod
    def num_eval_processes(cls, args: Args):
        return cls._num_eval_processes(args.num_processes, args.num_test_envs)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def tokenizer(pretrained_model):
        return GPT2Tokenizer.from_pretrained(pretrained_model)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def train_test_split(
        names: Tuple[str], num_test: int, rng: np.random.Generator
    ) -> TrainTest[Set[str]]:
        test_set = (
            set(rng.choice(list(names), size=num_test, replace=False))
            if num_test
            else set()
        )
        train_set = set(names) - test_set
        return TrainTest(train=train_set, test=test_set)

    # noinspection PyMethodOverriding
    @classmethod
    def _make_vec_envs(
        cls,
        data_path: str,
        num_envs: int,
        pretrained_model: str,
        models: Optional[str],
        names: Optional[str],
        num_processes: int,
        num_test_envs: int,
        num_test_names: int,
        render: bool,
        seed: int,
        sync_envs: bool,
        test: bool,
        **kwargs,
    ):
        assert num_envs >= num_processes
        if test:
            num_processes = cls._num_eval_processes(num_processes, num_test_envs)
            num_envs = num_test_envs

        if render:
            num_envs = 1

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

        # mapping = {}
        # for subdir in data_path.iterdir():
        #     with Path(subdir, "meta.json").open() as f:
        #         meta = json.load(f)
        #     name = meta["model_cat"]
        #     mapping[subdir.name] = name

        if names:
            names: Set[str] = set(names.split(","))
        if models:
            models: Set[str] = set(models.split(","))
        urdfs = list(get_urdfs(data_path, models, names))
        names: List[str] = [urdf.name for urdf in urdfs]
        longest_mission = max(names, key=len)
        rng = np.random.default_rng(seed=seed)
        names: TrainTest[Set[str]] = cls.train_test_split(
            tuple(names), num_test_names, rng
        )
        names: Set[str] = names.test if test else names.train
        assert len(names) > 1
        urdfs = [u for u in urdfs if u.name in names]

        def get_pairs():
            while True:
                rng.shuffle(urdfs)
                for urdf in urdfs:
                    opposites = [u for u in urdfs if u.name != urdf.name]
                    opposite = opposites[rng.choice(len(opposites))]
                    yield urdf, opposite

        pairs = list(itertools.islice(get_pairs(), num_envs))

        envs = [
            cls.make_env(
                longest_mission=longest_mission,
                rank=i,
                render=render,
                seed=seed + i,
                test=test,
                tokenizer=cls.tokenizer(pretrained_model=pretrained_model),
                urdfs=pairs[i],
                **kwargs,
            )
            for i in range(num_envs)
        ]

        if len(envs) == 1 or sync_envs or render:
            envs = DummyVecEnv(envs, num_processes)
        else:
            envs = SubprocVecEnv(envs, num_processes)
        return envs


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
