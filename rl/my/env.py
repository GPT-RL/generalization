import string
import typing
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, TypeVar

import attr
import gym
import habitat
import habitat_sim
import numpy as np
from art import text2art
from colors import color
from gym import Space
from gym.utils import seeding
from habitat import Config, Dataset
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from tap import Tap

EPISODE_SUCCESS = "episode success"
EXCLUDED = "excluded"
DESCRIPTION = "description"
NAME = "name"
PATH = "path"
OBJECT = "object"


class Args(Tap):
    scene: Optional[str] = None


@attr.s(auto_attribs=True, slots=True)
class MoveBackwardSpec:
    backward_amount: float


@habitat_sim.registry.register_move_fn(body_action=True)
class MoveBackward(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: MoveBackwardSpec
    ):
        backward_ax = (
            np.array(scene_node.absolute_transformation().rotation_scaling())
            @ habitat_sim.geo.BACK
        )
        scene_node.translate_local(backward_ax * actuation_spec.backward_amount)


T = TypeVar("T")  # Declare type variable


@dataclass
class Obs:
    image: T
    mission: T

    def to_space(self):
        return gym.spaces.Dict(**asdict(self))

    def to_obs(self, observation_space: Space = None, check: bool = False):
        obs = asdict(self)
        if observation_space and check:
            assert observation_space.contains(obs), (observation_space, obs)
        return obs


class PlacementError(RuntimeError):
    pass


class String(gym.Space):
    def sample(self):
        return "".join(self.np_random.choice(string.ascii_letters) for _ in range(10))

    def contains(self, x):
        return isinstance(x, str)


class StringTuple(gym.Space):
    def sample(self):
        return []

    def contains(self, x):
        return isinstance(x, tuple) and all([isinstance(y, str) for y in x])


# https://github.com/facebookresearch/habitat-lab/blob/main/examples/new_actions.py#L154


@attr.s(auto_attribs=True, slots=True)
class NoisyStrafeActuationSpec:
    move_amount: float
    # Classic strafing is to move perpendicular (90 deg) to the forward direction
    strafe_angle: float = 90.0
    noise_amount: float = 0.05


def _strafe_impl(
    scene_node: habitat_sim.SceneNode,
    move_amount: float,
    strafe_angle: float,
    noise_amount: float,
):
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    strafe_angle = np.deg2rad(strafe_angle)
    strafe_angle = np.random.uniform(
        (1 - noise_amount) * strafe_angle, (1 + noise_amount) * strafe_angle
    )

    rotation = habitat_sim.utils.quat_from_angle_axis(strafe_angle, habitat_sim.geo.UP)
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)

    move_amount = np.random.uniform(
        (1 - noise_amount) * move_amount, (1 + noise_amount) * move_amount
    )
    scene_node.translate_local(move_ax * move_amount)


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyStrafeLeft(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyStrafeActuationSpec,
    ):
        _strafe_impl(
            scene_node,
            actuation_spec.move_amount,
            actuation_spec.strafe_angle,
            actuation_spec.noise_amount,
        )


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyStrafeRight(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyStrafeActuationSpec,
    ):
        _strafe_impl(
            scene_node,
            actuation_spec.move_amount,
            -actuation_spec.strafe_angle,
            actuation_spec.noise_amount,
        )


@habitat.registry.register_action_space_configuration
class NoNoiseStrafe(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.STRAFE_LEFT] = habitat_sim.ActionSpec(
            "noisy_strafe_left",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.0),
        )
        config[HabitatSimActions.STRAFE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_strafe_right",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.0),
        )

        return config


@habitat.registry.register_action_space_configuration
class NoiseStrafe(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.STRAFE_LEFT] = habitat_sim.ActionSpec(
            "noisy_strafe_left",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.05),
        )
        config[HabitatSimActions.STRAFE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_strafe_right",
            NoisyStrafeActuationSpec(0.25, noise_amount=0.05),
        )

        return config


@habitat.registry.register_task_action
class StrafeLeft(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_left"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.STRAFE_LEFT)


@habitat.registry.register_task_action
class StrafeRight(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "strafe_right"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.STRAFE_RIGHT)


class Env(habitat.Env, gym.Env):
    def __init__(
        self,
        attributes: typing.Optional[typing.Dict[str, typing.Tuple[str, ...]]],
        config: Config,
        dataset: Optional[Dataset] = None,
        ids_to_objects: Optional[dict] = None,
        scene: Optional[str] = None,
        size: Optional[int] = None,
    ):
        HabitatSimActions.extend_action_space("STRAFE_LEFT")
        HabitatSimActions.extend_action_space("STRAFE_RIGHT")

        config.defrost()

        config.TASK.POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS + [
            "STRAFE_LEFT",
            "STRAFE_RIGHT",
        ]
        config.TASK.ACTIONS.STRAFE_LEFT = habitat.config.Config()
        config.TASK.ACTIONS.STRAFE_LEFT.TYPE = "StrafeLeft"
        config.TASK.ACTIONS.STRAFE_RIGHT = habitat.config.Config()
        config.TASK.ACTIONS.STRAFE_RIGHT.TYPE = "StrafeRight"
        config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoiseStrafe"

        scenes_dir = Path(config.DATASET.SCENES_DIR).expanduser()
        config.DATASET.SCENES_DIR = str(scenes_dir)
        data_path = Path(config.DATASET.DATA_PATH).expanduser()
        if scene is not None:
            assert scene.endswith(".json.gz"), scene
            data_path = data_path.with_name(scene)
        config.DATASET.DATA_PATH = str(data_path)
        if size:
            config.SIMULATOR.RGB_SENSOR.WIDTH = size
            config.SIMULATOR.RGB_SENSOR.HEIGHT = size
            config.SIMULATOR.DEPTH_SENSOR.WIDTH = size
            config.SIMULATOR.DEPTH_SENSOR.HEIGHT = size
            config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = size
            config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = size
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
            4,  # door
            63,  # towel
            76,  # sink
            83,  # mirror
            89,  # picture
            98,  # door
        ]
        self.np_random, seed = seeding.np_random(0)
        super().__init__(config, dataset)

        def get_object_ids():
            for level in self.sim.semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
                        if obj.category.name() not in excluded:
                            _, _, obj_id = obj.id.split("_")
                            obj_id = int(obj_id)
                            if obj_id not in excluded:
                                yield obj_id, obj.category.name()

        self.ids_to_object = (
            dict(get_object_ids()) if ids_to_objects is None else ids_to_objects
        )
        self.object_to_ids = defaultdict(list)

        for k, v in self.ids_to_object.items():
            self.object_to_ids[v].append(k)
        self.objects = sorted(self.object_to_ids.keys())
        self.features = (
            {o: [o] for o in self.objects} if attributes is None else attributes
        )

        self.action = None
        self.observations = None

    def _episode_success(self, observations: Observations) -> bool:
        try:
            if self.task.is_episode_active:
                return False
        except AttributeError:
            return False
        overlay = self.objective_overlay(observations)
        np_sum = np.sum(overlay)
        return np_sum > 50

    def objective_overlay(self, observations):
        objective_ids = self.object_to_ids[self.objective]
        objective_ids = np.array(objective_ids).reshape((-1, 1, 1))
        depth = observations["depth"].copy()
        semantic = observations["semantic"].copy()
        is_objective = semantic == objective_ids
        in_range = depth.squeeze(-1) < self._config.TASK.SUCCESS.SUCCESS_DISTANCE
        return in_range & is_objective.any(0)

    @staticmethod
    def ascii_of_image(image: np.ndarray):
        for row in image:
            yield "".join([(color("██", tuple(rgb.astype(int)))) for rgb in row])

    def get_done(self, observations: Observations) -> bool:
        done = False
        if self.episode_over or self._episode_success(observations):
            done = True
        return done

    def get_info(self, observations: Observations) -> typing.Dict[str, typing.Any]:
        i = dict({OBJECT: self.features[self.objective]})
        if self.get_done(observations):
            i.update({EPISODE_SUCCESS: self._episode_success(observations)})
        return i

    def get_reward(self, observations: Observations) -> float:
        if self._episode_success(observations):
            return self._max_reward - 0.2 * (
                self._elapsed_steps / self._max_episode_steps
            )
        return 0

    def get_reward_range(self):
        return self._slack_reward, self._max_reward

    def render(self, mode="rgb", pause=True) -> np.ndarray:
        if mode == "ascii":
            rgb = self.observations["rgb"]
            rgb_strings = list(self.ascii_of_image(rgb))
            semantic = self.observations["semantic"]
            semantic = np.expand_dims(semantic, -1)
            rgb, semantic = np.broadcast_arrays(rgb, semantic)
            # overlay = self.objective_overlay(self.observations)
            # overlay = np.expand_dims(overlay, -1)
            # rgb, overlay = np.broadcast_arrays(rgb, overlay)
            # overlay_strings = list(self.ascii_of_image(255 * overlay))
            semantic_strings = list(self.ascii_of_image(semantic))
            for rgb_string, semantic_string in zip(rgb_strings, semantic_strings):
                print(f"{rgb_string}{semantic_string}")
            # for rgb_string, overlay_string in zip(rgb_strings, overlay_strings):
            #     print(f"{rgb_string}{overlay_string}")
            # if overlay.sum() > 50:
            #     breakpoint()
            subtitle = str(self.objective)
            if self.action is not None:
                action_str = (
                    self.task.get_action_name(self.action)
                    if isinstance(self.action, int)
                    else self.action
                )
                subtitle += f", {action_str}"
            subtitle += f", reward={self.get_reward(self.observations)}"
            if self.get_done(self.observations):
                subtitle += ", done"
            print(text2art(subtitle.swapcase(), font="com_sen"))
            if pause:
                input("Press enter to continue.")

        else:
            return super().render(mode=mode)

    def reset(self) -> dict:
        idx = self.np_random.choice(len(self.ids_to_object))
        self.obj_id, self.objective = list(self.ids_to_object.items())[idx]
        self.observations = super().reset()
        return self.observations

    def seed(self, seed: int) -> None:
        self.np_random, self._seed = seeding.np_random(seed)

    def step(self, action, *args, **kwargs) -> typing.Tuple[dict, float, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """
        self.action = action
        self.observations = observations = super().step(*args, action=action, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        return observations, reward, done, info
