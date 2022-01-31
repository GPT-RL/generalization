import abc
from dataclasses import astuple, dataclass, replace
from functools import lru_cache
from itertools import chain, cycle, islice
from typing import TypeVar

import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple
from pybullet_env import Observation
from transformers import GPT2Tokenizer

T = TypeVar("T")  # Declare type variable


@dataclass
class TrainTest:
    train: list
    test: list


class MissionWrapper(gym.Wrapper, abc.ABC):
    def __init__(self, env):
        self._mission = None
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._mission = self.change_mission(observation["mission"])
        observation["mission"] = self._mission
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation["mission"] = self._mission
        return observation, reward, done, info

    def render(self, mode="human", pause=True, **kwargs):
        self.env.render(pause=False)
        print(self._mission)
        self.env.pause(pause)

    def change_mission(self, mission: str) -> str:
        raise NotImplementedError


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = Observation(*self.observation_space.spaces)
        self.original_observation_space = Tuple(
            astuple(Observation(*self.observation_space.spaces))
        )

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
    def __init__(self, env, tokenizer: GPT2Tokenizer, longest_mission: str):
        self.tokenizer: GPT2Tokenizer = tokenizer
        encoded = tokenizer.encode(longest_mission)
        super().__init__(env)
        spaces = Observation(*self.observation_space.spaces)
        self.observation_space = Tuple(
            astuple(
                replace(
                    spaces,
                    mission=MultiDiscrete([tokenizer.eos_token_id for _ in encoded]),
                )
            )
        )
        self.length = len(encoded)

    def observation(self, observation):
        observation = Observation(*observation)
        mission = observation.mission
        tokenizer = self.tokenizer
        mission = self.new_mission(self.length, mission, tokenizer)
        return astuple(replace(observation, mission=mission))

    @staticmethod
    @lru_cache()
    def new_mission(length, mission, tokenizer):
        mission = tokenizer.encode(mission)
        eos = tokenizer.eos_token_id
        mission = np.array([*islice(chain(mission, cycle([eos])), length)])
        return mission
