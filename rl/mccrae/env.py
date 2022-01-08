from __future__ import annotations

import itertools
from dataclasses import astuple, dataclass, replace
from typing import Generator, Generic, NamedTuple, TypeVar

import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Discrete, MultiDiscrete
from transformers import BertTokenizer, GPT2Tokenizer

T = TypeVar("T")


@dataclass
class Obs(Generic[T]):
    image: T
    mission: T


class Step(NamedTuple):
    obs: Obs[np.ndarray]
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
            astuple(
                Obs[gym.Space](
                    image=Box(
                        low=0,
                        high=1,
                        shape=[self.room_size, self.room_size, self.d],
                    ),
                    mission=MultiDiscrete(nvec),
                )
            )
        )
        self.action_space = Discrete(5)
        self._render = None

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
        invalid = True
        while invalid:
            goal, distractor = self.rng.choice(
                len(self.concepts), size=2, replace=False
            )
            goal_features = self.features[goal]
            distractor_features = self.features[distractor]
            invalid = np.array_equal(goal_features, distractor_features)
        mission = self.concepts[goal]
        goal_pos_x, goal_pos_y = goal_pos
        room_array = np.zeros((self.room_size, self.room_size, self.d))
        room_array[goal_pos_x, goal_pos_y, :-1] = goal_features
        distractor_pos_x, distractor_pos_y = distractor_pos
        room_array[distractor_pos_x, distractor_pos_y, :-1] = distractor_features
        deltas = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }
        action = None
        done = False
        reward = 0
        while True:
            room_with_agent = np.copy(room_array)
            agent_pos_x, agent_pos_y = agent_pos
            room_with_agent[agent_pos_x, agent_pos_y, -1] = 1

            def row_string(row: int):
                yield "|"
                for col in range(self.room_size):
                    if (row, col) == tuple(agent_pos):
                        yield "a"
                    elif (row, col) == goal_pos:
                        yield "g"
                    elif (row, col) == distractor_pos:
                        yield "d"
                    else:
                        yield " "
                yield "|"

            def strings():
                horizontal_wall = f"+{'-' * self.room_size}+"
                yield horizontal_wall
                for row in range(self.room_size):
                    yield "".join(row_string(row))
                yield horizontal_wall

            def string():
                return "\n".join(strings())

            def render():
                print(room_with_agent.transpose(2, 0, 1))
                print(string())
                print("action:", None if action is None else deltas.get(action, "done"))
                print("reward:", reward)

            self._render = render

            action = yield Step(
                astuple(Obs[np.ndarray](image=room_with_agent, mission=mission)),
                reward,
                done,
                {},
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

    def render(self, mode="human", pause=True):
        self._render()
        if pause:
            input("Press enter to continue.")


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
            astuple(Obs[gym.Space](*self.observation_space.spaces))
        )

        self.observation_space = Box(
            shape=[sum(map(get_size, self.observation_space.spaces))],
            low=-np.inf,
            high=np.inf,
        )

    def observation(self, obs: tuple):
        obs = Obs(*obs)
        return np.concatenate(astuple(replace(obs, image=obs.image.flatten())))
