from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Generator, Generic, NamedTuple, TypeVar

import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Discrete, MultiDiscrete
from transformers import BertTokenizer, GPT2Tokenizer

I = TypeVar("I")
M = TypeVar("M")


class Obs(NamedTuple, Generic[I, M]):
    image: I
    mission: M


class Step(NamedTuple):
    obs: Obs[np.ndarray, np.ndarray]
    reward: float
    done: bool
    info: dict


@dataclass
class Env(gym.Env):
    concepts: np.ndarray
    features: np.ndarray
    max_token_id: int
    room_size: int
    seed: int

    def __post_init__(self):
        self.iterator = None
        self.rng = np.random.default_rng(seed=self.seed)
        self.d = self.features.shape[-1] + 1
        nvec = np.ones(self.concepts.shape[-1]) * self.max_token_id
        self.observation_space = spaces.Tuple(
            Obs(
                image=Box(
                    low=0,
                    high=1,
                    shape=[self.room_size, self.room_size, self.d],
                ),
                mission=MultiDiscrete(nvec),
            )
        )
        self.action_space = Discrete(5)

    def generator(
        self,
    ) -> Generator[Step, int, None]:
        positions = list(
            itertools.product(range(self.room_size), range(self.room_size))
        )
        agent, goal, distractor = self.rng.choice(len(positions), size=3, replace=False)
        agent_pos = np.array(positions[agent])
        goal_pos = positions[goal]
        distractor_pos = positions[distractor]
        room_array = np.zeros((self.room_size, self.room_size, self.d))
        goal, distractor = self.rng.choice(len(self.concepts), size=2, replace=False)
        mission = self.concepts[goal]
        goal_features = self.features[goal]
        distractor_features = self.features[goal]
        goal_pos_x, goal_pos_y = goal_pos
        room_array[goal_pos_x, goal_pos_y, :-1] = goal_features
        distractor_pos_x, distractor_pos_y = distractor_pos
        room_array[distractor_pos_x, distractor_pos_y, :-1] = distractor_features
        deltas = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }
        done = False
        reward = 0
        while True:
            agent_pos_x, agent_pos_y = agent_pos
            room_array[agent_pos_x, agent_pos_y, -1] = 1
            action = yield Step(
                Obs(image=room_array, mission=mission), reward, done, {}
            )
            if action in deltas:
                delta = np.array(deltas[action])
                agent_pos += delta
                agent_pos = np.clip(agent_pos, a_min=0, a_max=self.room_size - 1)
            else:
                assert action == len(deltas)
                reward = tuple(agent_pos) == goal_pos
                done = True

    def step(self, action: np.ndarray):
        return self.iterator.send(int(action))

    def reset(self):
        self.iterator = self.generator()
        return next(self.iterator).obs

    def render(self, mode="human"):
        pass


def make_tokenizer(pretrained_model):
    if "gpt" in pretrained_model:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    elif "bert" in pretrained_model:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    else:
        raise RuntimeError(f"Invalid model name: {pretrained_model}")
    return tokenizer


def get_size(space: gym.Space):
    if isinstance(space, Box):
        return np.prod(space.shape)
    elif isinstance(space, MultiDiscrete):
        return space.nvec.size
    elif isinstance(space, Discrete):
        return 1


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_observation_space = spaces.Tuple(
            Obs(*self.observation_space.spaces)
        )

        self.observation_space = Box(
            shape=[sum(map(get_size, self.observation_space.spaces))],
            low=-np.inf,
            high=np.inf,
        )

    def observation(self, obs: Obs):
        return np.concatenate(Obs(image=obs.image.flatten(), mission=obs.mission))
