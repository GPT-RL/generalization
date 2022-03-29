import itertools
import os
import typing
from collections import defaultdict
from typing import Optional

import gym
import habitat
import numpy as np
from colors import color
from gym.utils import seeding
from habitat import Config, Dataset
from habitat.core.simulator import Observations


class Env(habitat.Env, gym.Env):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        config.defrost()
        config.DATASET.DATA_PATH = os.path.expanduser(config.DATASET.DATA_PATH)
        config.DATASET.SCENES_DIR = os.path.expanduser(config.DATASET.SCENES_DIR)
        config.freeze()
        self._slack_reward = 0
        self._max_reward = 1
        self.reward_range = self.get_reward_range()
        excluded = [
            "furniture",
            "objects",
            "void",
            "",
            "misc",
            "floor",
            "ceiling",
            "wall",
        ]
        self.np_random, seed = seeding.np_random(0)
        super().__init__(config, dataset)

        def get_object_ids():
            for level in self.sim.semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
                        if obj.category.name() not in excluded:
                            _, _, obj_id = obj.id.split("_")
                            yield int(obj_id), obj.category.name()

        self.ids_to_object = dict(get_object_ids())
        self.object_to_ids = defaultdict(list)
        for k, v in self.ids_to_object.items():
            self.object_to_ids[v].append(k)
        self.objects = sorted(self.object_to_ids.keys())
        self.features = {o: [o] for o in self.objects}

    def _episode_success(self, observations: Observations) -> bool:

        objective_ids = self.object_to_ids[self.objective]
        objective_ids = np.array(objective_ids).reshape((-1, 1, 1))
        depth = observations["depth"].copy()
        semantic = observations["semantic"].copy()
        expanded = np.expand_dims(semantic, 0)
        is_objective = expanded == objective_ids
        is_objective = is_objective.any(0)
        objective_in_range = depth.squeeze(-1)[is_objective]
        if not objective_in_range.size:
            return False
        objective_in_range = objective_in_range.min()

        if self.task.is_episode_active:
            return False
        return objective_in_range < self._config.TASK.SUCCESS.SUCCESS_DISTANCE

    @staticmethod
    def ascii_of_image(image: np.ndarray):
        def rows():
            for row in image:
                yield "".join([(color("██", tuple(rgb.astype(int)))) for rgb in row])

        return "\n".join(rows())

    def get_done(self, observations: Observations) -> bool:
        done = False
        if self.episode_over or self._episode_success(observations):
            done = True
        return done

    def get_info(self, observations: Observations) -> typing.Dict[str, typing.Any]:
        i = dict(mission=self.features[self.objective])
        if self.get_done(observations):
            i.update(success=self._episode_success(observations))
        return i

    def get_reward(self, observations: Observations) -> float:
        if self._episode_success(observations):
            return self._max_reward - 0.2 * (
                self._elapsed_steps / self._max_episode_steps
            )
        return 0

    def get_reward_range(self):
        return self._slack_reward, self._max_reward

    def render(self, mode="rgb") -> np.ndarray:
        if mode == "ascii":
            raise NotImplementedError()
        else:
            return super().render(mode=mode)

    def reset(self) -> dict:
        idx = self.np_random.choice(len(self.objects))
        self.objective = self.objects[idx]
        return super().reset()

    def seed(self, seed: int) -> None:
        self.np_random, seed = seeding.np_random(seed)

    def step(self, *args, **kwargs) -> typing.Tuple[dict, float, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = super().step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        return observations, reward, done, info


if __name__ == "__main__":
    env = Env(config=habitat.get_config("objectnav_mp3d.yaml"))
    for i in itertools.count():
        print(i)
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
