import abc
import sys
import typing
from dataclasses import astuple, dataclass, replace
from functools import lru_cache
from typing import Dict, Generic, List, TypeVar

import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from my.env import Obs
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

T = TypeVar("T")  # Declare type variable

EPISODE_SUCCESS = "episode success"
FAIL_SEED_SUCCESS = "fail seed success"
FAIL_SEED_USAGE = "fail seed usage"
NUM_FAIL_SEEDS = "number of fail seeds"


@dataclass
class TrainTest(Generic[T]):
    train: T
    test: T


class DirectionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            spaces=dict(
                **self.observation_space.spaces,
                direction=Discrete(4),
            )
        )


class ImageNormalizerWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        observation = Obs(**observation)
        return replace(observation, image=observation.image / 255).to_obs(
            self.observation_space
        )


class MissionWrapper(gym.Wrapper, abc.ABC):
    def __init__(self, env):
        self._mission = None
        super().__init__(env)

    def reset(self, **kwargs):
        observation = Obs(**self.env.reset(**kwargs))
        self._mission = self.change_mission(observation.mission)
        return replace(observation, mission=self._mission).to_obs(
            self.observation_space
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = replace(Obs(**observation), mission=self._mission)
        return observation.to_obs(self.observation_space), reward, done, info

    def render(self, mode="human", pause=True, **kwargs):
        self.env.render(pause=False)
        print(self._mission)
        self.env.pause(pause)

    def change_mission(self, mission: str) -> str:
        raise NotImplementedError


class FeatureWrapper(MissionWrapper):
    def __init__(self, env: gym.Env, features: Dict[str, List[str]]):
        super().__init__(env)
        self.features = features
        observation_space = Obs(**self.observation_space.spaces)
        self.observation_space = replace(
            observation_space, mission=StringTuple()
        ).to_space()

    def change_mission(self, mission: str) -> List[str]:
        return self.features[mission]


class RenderWrapper(gym.Wrapper):
    def __init__(self, env, mode="human"):
        self.mode = mode
        super().__init__(env)

    def step(self, action):
        self.render(mode=self.mode, pause=False)
        s, r, t, i = super().step(action)
        if t:
            self.render(mode=self.mode, pause=r == 0)
        return s, r, t, i


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = Obs(**self.observation_space.spaces)
        self.original_observation_space = spaces.to_space()

        def sizes():
            for space in astuple(spaces):
                if isinstance(space, Box):
                    yield np.prod(space.shape)
                elif isinstance(space, MultiDiscrete):
                    yield space.nvec.size
                elif isinstance(space, Discrete):
                    yield 1

        self.observation_space = Box(
            shape=[sum(sizes())],
            low=-np.inf,
            high=np.inf,
        )

    def observation(self, observation):
        obs = np.concatenate(
            astuple(Obs(**{k: v.flatten() for k, v in observation.items()}))
        )
        # assert self.observation_space.contains(obs)
        return obs


class SuccessWrapper(gym.Wrapper):
    def step(self, action):
        s, r, t, i = super().step(action)
        if t:
            i.update({EPISODE_SUCCESS: r > 0})
        return s, r, t, i


class FailureReplayWrapper(SuccessWrapper):
    def __init__(self, env, seed: int, tgt_success_prob: float):
        self.tgt_success_prob = tgt_success_prob
        self.regular_successes = []
        self.fail_seed_successes = []
        self.fail_seeds = []
        self.rng = np.random.default_rng(seed=seed)
        super().__init__(env)

    def reset(self, **kwargs):
        regular_success_prob = float(np.mean(self.regular_successes))
        fail_seed_success_prob = (
            np.mean(self.fail_seed_successes) if self.fail_seed_successes else 0
        )
        if (
            self.regular_successes
            and self.fail_seeds
            and regular_success_prob != fail_seed_success_prob
        ):
            use_fail_seeds_prob = (self.tgt_success_prob - regular_success_prob) / (
                fail_seed_success_prob - regular_success_prob
            )
        else:
            use_fail_seeds_prob = 0
        self.using_fail_seed = self.rng.random() < use_fail_seeds_prob
        if self.using_fail_seed:
            i = self.rng.choice(len(self.fail_seeds))
            self.current_seed = self.fail_seeds.pop(i)
        else:
            self.current_seed = self.rng.choice(sys.maxsize)
        self.env.seed(self.current_seed)
        return super().reset(**kwargs)

    def step(self, action):
        s, r, t, i = super().step(action)
        if t:
            i.update({NUM_FAIL_SEEDS: len(self.fail_seeds)})
            if self.using_fail_seed:
                success = i.pop(EPISODE_SUCCESS)
                i.update({FAIL_SEED_SUCCESS: success})
                if not success and len(self.fail_seeds) < 100:
                    self.fail_seed_successes += [success]
            else:
                success = i[EPISODE_SUCCESS]
                self.regular_successes += [success]
                if not success and len(self.fail_seeds) < 100:
                    self.fail_seeds += [self.current_seed]
        return s, r, t, i


class StringTuple(gym.Space):
    def sample(self):
        return []

    def contains(self, x):
        return isinstance(x, tuple) and all([isinstance(y, str) for y in x])


class TokenizerWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        all_missions: list,
        tokenizer: GPT2Tokenizer,
    ):
        self.tokenizer: GPT2Tokenizer = tokenizer
        ns, ds = zip(
            *[
                self.encode(tuple(m) if isinstance(m, list) else m, tokenizer).shape
                for m in all_missions
            ]
        )
        n = max(ns)
        d = max(ds)

        super().__init__(env)
        spaces = Obs(**self.observation_space.spaces)

        self.obs_spaces = replace(
            spaces,
            mission=MultiDiscrete((1 + tokenizer.eos_token_id) * np.ones((n, d))),
        )
        self.observation_space = self.obs_spaces.to_space()

    @staticmethod
    def encode(
        mission: typing.Union[typing.Tuple[str], str],
        tokenizer: GPT2Tokenizer,
    ):
        if isinstance(mission, tuple):

            def get_tokens():
                for w in mission:
                    encoded = tokenizer.encode(w, return_tensors="pt")
                    encoded = typing.cast(Tensor, encoded)
                    yield encoded.T

            tokens = list(get_tokens())
            padded = (
                pad_sequence(tokens, padding_value=tokenizer.eos_token_id).squeeze(-1).T
            )
            return padded.numpy()
        elif isinstance(mission, str):
            return tokenizer.encode(mission, return_tensors="np")
        else:
            raise RuntimeError()

    @classmethod
    @lru_cache()
    def new_mission(
        cls,
        mission: typing.Union[typing.Tuple[str], str],
        mission_shape: typing.Tuple[int, int],
        tokenizer: GPT2Tokenizer,
    ):
        encoded = cls.encode(mission, tokenizer)
        n1, d1 = encoded.shape
        n2, d2 = mission_shape

        assert n2 >= n1 and d2 >= d1
        padded = np.pad(
            encoded,
            [(0, n2 - n1), (0, d2 - d1)],
            constant_values=tokenizer.eos_token_id,
        )
        return padded.reshape(mission_shape)

    def observation(self, observation):
        observation = Obs(**observation)
        mission = observation.mission
        if isinstance(mission, list):
            mission = tuple(mission)
        mission = self.new_mission(
            mission=mission,
            mission_shape=tuple(self.obs_spaces.mission.nvec.shape),
            tokenizer=self.tokenizer,
        )
        return replace(observation, mission=mission).to_obs(self.observation_space)
