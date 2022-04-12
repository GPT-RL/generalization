from __future__ import annotations

import sys
import typing
from dataclasses import astuple, dataclass, replace
from typing import Generic, List, TypeVar

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiDiscrete
from my.env import EPISODE_SUCCESS, Obs, String
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import CLIPProcessor, GPT2Tokenizer

T = TypeVar("T")  # Declare type variable

FAIL_SEED_SUCCESS = "fail seed success"
FAIL_SEED_USAGE = "fail seed usage"
NUM_FAIL_SEEDS = "number of fail seeds"


@dataclass
class TrainTest(Generic[T]):
    train: T
    test: T


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


class ActionWrapper(gym.ActionWrapper):
    def reverse_action(self, action):
        raise NotImplementedError()

    def action(self, action):
        return action.item()


class ActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = Discrete(len(env.action_space))


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        spaces = self.observation_space.spaces
        self.observation_space = Obs(image=spaces["rgb"], mission=String()).to_space()

    def observation(self, observation):
        mission = self.env.features[self.env.objective]
        return Obs(image=observation["rgb"], mission=mission).to_obs(
            self.observation_space
        )


class RenderWrapper(gym.Wrapper):
    def __init__(self, env, mode="human"):
        self.mode = mode
        super().__init__(env)

    def step(self, action):
        self.render(mode=self.mode, pause=False)
        s, r, t, i = super().step(action)
        if t:
            self.render(mode=self.mode, pause=True)
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
            spaces, mission=Box(low=encodings.min(), high=encodings.max(), shape=(n, d))
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

        encodings = [
            embedding_for_mission(m).numpy().mean(0, keepdims=True)
            for m in all_missions
        ]
        array = np.stack(encodings)
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
            return (
                pad_sequence(tokens, padding_value=tokenizer.eos_token_id).squeeze(-1).T
            )

        encodings = [encode(m) for m in all_missions]
        ns, ds = zip(*[e.shape for e in encodings])
        n, d = max(ns), max(ds)
        padded = [
            np.pad(
                e,
                ((0, n - e.shape[0]), (0, d - e.shape[1])),
                mode="constant",
                constant_values=self.tokenizer.eos_token_id,
            )
            for e in encodings
        ]
        array = np.stack(padded)
        _, array_inverse = np.unique(array, return_inverse=True)
        array = array_inverse.reshape(array.shape)
        super().__init__(env, all_missions, array)


class TransposeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = Obs(**self.observation_space.spaces)
        self.observation_space = replace(
            spaces,
            image=Box(
                low=spaces.image.low.transpose(2, 0, 1),
                high=spaces.image.high.transpose(2, 0, 1),
            ),
        ).to_space()

    def observation(self, observation):
        obs = Obs(**observation)
        return replace(obs, image=obs.image.transpose(2, 0, 1)).to_obs()
