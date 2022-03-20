from __future__ import annotations

import abc
import sys
import typing
from collections import deque
from dataclasses import astuple, dataclass, replace
from typing import Dict, Generic, List, TypeVar

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiDiscrete
from my.env import PAIR, Obs, StringTuple
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import CLIPProcessor, GPT2Tokenizer
from utils import softmax

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


class CLIPProcessorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, processor: CLIPProcessor):
        self.processor = processor
        super().__init__(env)
        self.observation_space.spaces.update(
            image=Box(low=0, high=255, shape=[3, 224, 224])
        )

    def observation(self, observation):
        obs = Obs(**observation)
        image = self.processor(images=obs.image, return_tensors="np", padding=True)
        return replace(obs, image=image["pixel_values"]).to_obs(self.observation_space)


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


class PairsSelectorWrapper(SuccessWrapper):
    def __init__(self, env, objects: List[str], seed: int, temp: float):
        self.temp = temp
        self.counter = {
            (o1, o2): deque([0], maxlen=100)
            for o1 in objects
            for o2 in objects
            if o1 != o2
        }
        self.rng = np.random.default_rng(seed=seed)
        super().__init__(env)

    def reset(self, **kwargs):
        pairs, deques = zip(*self.counter.items())
        averages = np.array([np.mean(d) for d in deques])
        averages = np.maximum(averages, 1e-5)
        p = softmax(self.temp / averages)
        mesh_names = self.rng.choice(pairs, p=p)
        return super().reset(**kwargs, mesh_names=mesh_names)

    def step(self, action):
        s, r, t, i = super().step(action)
        if t:
            self.counter[i[PAIR]].append(r)
        return s, r, t, i


class FailureReplayWrapper(SuccessWrapper):
    def __init__(self, env, seed: int, tgt_success_prob: float = 0.5):
        self.tgt_success_prob = tgt_success_prob
        self.successes = []
        self.fail_seeds = []
        self.rng = np.random.default_rng(seed=seed)
        super().__init__(env)

    def reset(self, **kwargs):
        if self.successes and self.fail_seeds:
            prior_success_prob = float(np.mean(self.successes))
            no_fail_seed_prob = self.tgt_success_prob / max(prior_success_prob, 1e-5)
            use_fail_seeds_prob = 1 - no_fail_seed_prob
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
            else:
                success = i[EPISODE_SUCCESS]
                self.successes += [success]
                if not success and len(self.fail_seeds) < 100:
                    self.fail_seeds += [self.current_seed]
        return s, r, t, i


class MissionPreprocessor(gym.ObservationWrapper):
    def __init__(self, env, all_missions: List[tuple], encodings: np.ndarray):
        self.encodings = {m: e for m, e in zip(all_missions, encodings)}
        super().__init__(env)
        spaces = Obs(**self.observation_space.spaces)

        _, n, d = encodings.shape
        self.obs_spaces = replace(
            spaces, mission=MultiDiscrete(encodings.max(initial=0) * np.ones((n, d)))
        )
        self.observation_space = self.obs_spaces.to_space()

    def observation(self, observation):
        observation = Obs(**observation)
        mission = observation.mission
        if isinstance(mission, list):
            mission = tuple(mission)
        mission = self.encodings[mission]
        return replace(observation, mission=mission).to_obs(self.observation_space)


class GPT3Wrapper(MissionPreprocessor):
    def __init__(self, env: gym.Env, all_missions: list):
        self.embeddings = torch.load("embeddings.pt")
        all_missions = [m for m in all_missions]

        def embeddings_for_mission(mission: tuple):
            for w in mission:
                yield self.embeddings[w]

        def embedding_for_mission(mission: tuple):
            return torch.stack(list(embeddings_for_mission(mission)))

        encodings = [embedding_for_mission(m) for m in all_missions]
        tensor = torch.stack(encodings)
        array = tensor.numpy()
        super().__init__(env, all_missions, array)


class TokenizerWrapper(MissionPreprocessor):
    def __init__(
        self,
        env: gym.Env,
        all_missions: list,
        tokenizer: GPT2Tokenizer,
    ):
        self.tokenizer: GPT2Tokenizer = tokenizer

        def encode(mission: typing.Union[typing.Tuple[str], str]):
            def get_tokens():
                for w in mission:
                    encoded = tokenizer.encode(w, return_tensors="pt")
                    encoded = typing.cast(Tensor, encoded)
                    yield encoded.T

            tokens = list(get_tokens())
            return pad_sequence(tokens, padding_value=tokenizer.eos_token_id).squeeze(
                -1
            )

        encodings = [encode(m) for m in all_missions]
        padded = pad_sequence(encodings, padding_value=tokenizer.eos_token_id)
        permuted = padded.permute(1, 2, 0)
        array = permuted.numpy()
        _, array_inverse = np.unique(array, return_inverse=True)
        array = array_inverse.reshape(array.shape)
        super().__init__(env, all_missions, array)
