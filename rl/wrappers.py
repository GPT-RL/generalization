import abc
import typing
from dataclasses import astuple, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Generic, List, TypeVar

import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from pybullet_env import Observation
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

T = TypeVar("T")  # Declare type variable


@dataclass
class TrainTest(Generic[T]):
    train: T
    test: T


class ImageNormalizerWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        observation = Observation(*observation)
        return replace(observation, image=observation.image / 255).to_obs(
            self.observation_space
        )


class MissionWrapper(gym.Wrapper, abc.ABC):
    def __init__(self, env):
        self._mission = None
        super().__init__(env)

    def reset(self, **kwargs):
        observation = Observation(*self.env.reset(**kwargs))
        self._mission = self.change_mission(observation.mission)
        return replace(observation, mission=self._mission).to_obs(
            self.observation_space
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = replace(Observation(*observation), mission=self._mission)
        return observation.to_obs(self.observation_space), reward, done, info

    def render(self, mode="human", pause=True, **kwargs):
        self.env.render(pause=False)
        print(self._mission)
        self.env.pause(pause)

    def change_mission(self, mission: str) -> str:
        raise NotImplementedError


class StringTuple(gym.Space):
    def sample(self):
        return []

    def contains(self, x):
        return isinstance(x, tuple) and all([isinstance(y, str) for y in x])


class FeatureWrapper(MissionWrapper):
    def __init__(self, env: gym.Env, features: Dict[str, List[str]]):
        super().__init__(env)
        self.features = features
        observation_space = Observation(*self.observation_space.spaces)
        self.observation_space = replace(
            observation_space, mission=StringTuple()
        ).to_space()

    def change_mission(self, mission: str) -> List[str]:
        return self.features[mission]


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = Observation(*self.observation_space.spaces)
        self.original_observation_space = Observation(
            *self.observation_space.spaces
        ).to_space()

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
        obs = np.concatenate(astuple(Observation(*[x.flatten() for x in observation])))
        # assert self.observation_space.contains(obs)
        return obs


class TokenizerWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        all_missions: list,
        tokenizer: GPT2Tokenizer,
    ):
        self.tokenizer: GPT2Tokenizer = tokenizer
        ns, ds = zip(*[self.encode(m, tokenizer).shape for m in all_missions])
        n = max(ns)
        d = max(ds)

        super().__init__(env)
        spaces = Observation(*self.observation_space.spaces)

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
            tokens = [tokenizer.encode(w, return_tensors="pt").T for w in mission]
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
        observation = Observation(*observation)
        mission = observation.mission
        if isinstance(mission, list):
            mission = tuple(mission)
        mission = self.new_mission(
            mission=mission,
            mission_shape=tuple(self.obs_spaces.mission.nvec.shape),
            tokenizer=self.tokenizer,
        )
        return replace(observation, mission=mission).to_obs(self.observation_space)


class VideoRecorderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, path: Path):
        super().__init__(env)
        self.rec = VideoRecorder(env, path=str(path))

    def reset(self, **kwargs):
        s = super().reset()
        self.rec.capture_frame()
        print(Observation(*s).mission)
        return s

    def step(self, action):
        s, r, t, i = super().step(action)
        self.rec.capture_frame()
        return s, r, t, i

    def close(self):
        super().close()
        self.rec.close()
