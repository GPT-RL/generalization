from copy import deepcopy
from typing import Callable, List, Optional

import gym
import numpy as np
import stable_baselines3.common.vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs


class SubprocVecEnv(stable_baselines3.common.vec_env.SubprocVecEnv):
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        num_envs: int,
        start_method: Optional[str] = None,
    ):
        super().__init__(env_fns, start_method)
        self.in_use = list(range(num_envs))
        self.all = set(range(len(env_fns)))
        self.random = np.random.default_rng()

    def step_async(self, actions: np.ndarray) -> None:
        for r, action in zip(self.in_use, actions):
            self.remotes[r].send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [self.remotes[r].recv() for r in self.in_use]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        in_use = list(self.in_use)
        for i, (r, done, info) in enumerate(zip(in_use, dones, infos)):
            if done:
                # if i == 0:
                #     print(info["mission"])
                not_in_use = self.all - set(self.in_use)
                self.in_use[i] = self.random.choice([*not_in_use, r])

        return (
            _flatten_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def reset(self) -> VecEnvObs:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)[np.array(self.in_use)]


class DummyVecEnv(stable_baselines3.common.vec_env.DummyVecEnv):
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        num_envs: int,
    ):
        super().__init__(env_fns)
        self.in_use = list(range(num_envs))
        self.all = set(range(len(env_fns)))
        self.random = np.random.default_rng()

    def step_async(self, actions: np.ndarray) -> None:
        if self.actions is None:
            action = np.zeros_like(actions[0])
            self.actions = np.tile(action, (self.num_envs, 1))
        for env_idx, action in zip(self.in_use, actions):
            self.actions[env_idx] = action

    def step_wait(self) -> VecEnvStepReturn:
        in_use = list(self.in_use)
        for i, env_idx in enumerate(in_use):
            (
                obs,
                self.buf_rews[env_idx],
                self.buf_dones[env_idx],
                self.buf_infos[env_idx],
            ) = self.envs[env_idx].step(self.actions[env_idx])
            not_in_use = self.all - set(self.in_use)
            if self.buf_dones[env_idx]:
                # if i == 0:
                #     print(env_idx, self.buf_infos[env_idx]["mission"])
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
                self.in_use[i] = self.random.choice([*not_in_use, env_idx])
            self._save_obs(env_idx, obs)

        return (
            self._obs_from_buf()[in_use],
            np.copy(self.buf_rews)[in_use],
            np.copy(self.buf_dones)[in_use],
            [deepcopy(self.buf_infos)[i] for i in in_use],
        )

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()[self.in_use]
