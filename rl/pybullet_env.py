import json
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Generic, List, NamedTuple, Tuple, TypeVar, Union, cast

import gym.spaces as spaces
import gym.utils.seeding
import numpy as np
import PIL.Image
import pybullet as p
import torch
from gym.spaces import Box, MultiDiscrete
from pybullet_utils import bullet_client
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

CAMERA_DISTANCE = 0.1
CAMERA_PITCH = -10
CAMERA_YAW = 45

M = TypeVar("M")
I = TypeVar("I")


@dataclass
class Observation(Generic[M, I]):
    mission: M
    image: I


class Action(NamedTuple):
    turn: float = 0
    forward: float = 0
    done: bool = False
    take_picture: bool = False


class Actions(Enum):
    LEFT = Action(3, 0)
    RIGHT = Action(-3, 0)
    FORWARD = Action(0, 0.18)
    BACKWARD = Action(0, -0.18)
    DONE = Action(done=True)
    PICTURE = Action(take_picture=True)
    NO_OP = Action()


ACTIONS = [*Actions]


class URDF(NamedTuple):
    name: str
    path: Path
    z: float


@contextmanager
def suppress_stdout():
    """from https://stackoverflow.com/a/17954769/4176597"""
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def get_model_ids():
    with Path("model_ids.json").open() as f:
        return set(json.load(f))


def get_urdfs(path, model_ids):
    for subdir in path.iterdir():
        urdf = Path(subdir, "mobility.urdf")
        assert urdf.exists()
        with Path(subdir, "meta.json").open() as f:
            meta = json.load(f)
        with Path(subdir, "bounding_box.json").open() as f:
            box = json.load(f)
        _, _, z_min = box["min"]
        model_id = meta["model_id"]
        if model_id in model_ids:
            yield URDF(name=meta["model_cat"], path=urdf, z=-z_min)


@dataclass
class Env(gym.Env):
    tokenizer: GPT2Tokenizer
    urdfs: List[URDF]
    cameraYaw: float = CAMERA_YAW
    env_bounds: float = 5
    image_height: float = 64  # 72
    image_width: float = 64  # 96
    is_render: bool = False
    max_episode_steps: int = 200
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}
    model_name: str = "gpt2"
    reindex_tokens: bool = False

    def __post_init__(self):
        names, paths, zs = zip(*self.urdfs)
        self.missions = names
        self.seed(0)

        def tokens() -> Generator[torch.Tensor, None, None]:
            for k in names:
                encoded = self.tokenizer.encode(k, return_tensors="pt")
                tensor = cast(torch.Tensor, encoded)
                yield tensor.squeeze(0)

        padded = pad_sequence(
            list(tokens()),
            padding_value=self.tokenizer.eos_token_id,
        ).T.numpy()
        if self.reindex_tokens:
            _, indices = np.unique(padded, return_inverse=True)
            padded = indices.reshape(padded.shape)

        self.tokens = OrderedDict(zip(names, padded))

        max_padded = padded.max()
        nvec = np.ones_like(padded[0]) * (max_padded + 1)
        obs_spaces: Observation[MultiDiscrete, Box] = Observation(
            mission=MultiDiscrete(nvec),
            image=Box(
                low=0,
                high=255,
                shape=[self.image_height, self.image_width, 3],
            ),
        )
        self.observation_space = spaces.Tuple(astuple(obs_spaces))
        self.action_space = spaces.Discrete(5)

        self.iterator = None

        self.relativeChildPosition = [0, 0, 0]
        self.relativeChildOrientation = [0, 0, 0, 1]

        if self.is_render:
            with suppress_stdout():
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)
        else:
            with suppress_stdout():
                self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)

        sphereRadius = 0.2
        mass = 1
        visualShapeId = 2
        colSphereId = self._p.createCollisionShape(
            self._p.GEOM_SPHERE, radius=sphereRadius
        )
        self.mass = self._p.createMultiBody(
            mass, colSphereId, visualShapeId, [0, 0, 0.4]
        )

        self.mass_cid = self._p.createConstraint(
            self.mass,
            -1,
            -1,
            -1,
            self._p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.relativeChildPosition,
            self.relativeChildOrientation,
        )
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, False)

        # self._p.setGravity(0, 0, -10)
        halfExtents = [1.5 * self.env_bounds, 1.5 * self.env_bounds, 0.1]
        floor_collision = self._p.createCollisionShape(
            self._p.GEOM_BOX, halfExtents=halfExtents
        )
        floor_visual = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1, 1, 1, 0.5]
        )
        self._p.createMultiBody(0, floor_collision, floor_visual, [0, 0, -0.2])

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, False)

        self._p.setGravity(0, 0, -10)
        halfExtents = [1.5 * self.env_bounds, 1.5 * self.env_bounds, 0.1]
        floor_collision = self._p.createCollisionShape(
            self._p.GEOM_BOX, halfExtents=halfExtents
        )
        floor_visual = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1, 1, 1, 0.5]
        )
        self._p.createMultiBody(0, floor_collision, floor_visual, [0, 0, -0.2])

        self._p.setTimeStep(1)

        missions = []
        self.goals = goals = []
        urdfs = [
            self.urdfs[i]
            for i in self.random.choice(len(self.urdfs), size=2, replace=False)
        ]

        for base_position, urdf in zip(
            [
                [self.env_bounds / 3, self.env_bounds / 3, 0],
                [-self.env_bounds / 3, -self.env_bounds / 3, 0],
            ],
            urdfs,
        ):
            missions.append(urdf.name)
            base_position[-1] = urdf.z

            try:
                with suppress_stdout():
                    goal = self._p.loadURDF(
                        str(urdf.path), basePosition=base_position, useFixedBase=True
                    )
            except self._p.error:
                print(self._p.error)
                raise RuntimeError(f"Error while loading {urdf.path}")
            goals.append(goal)

            collisionFilterGroup = 0
            collisionFilterMask = 0
            self._p.setCollisionFilterGroupMask(
                goal, -1, collisionFilterGroup, collisionFilterMask
            )
            self._p.createConstraint(
                goal,
                -1,
                -1,
                -1,
                self._p.JOINT_FIXED,
                [1, 1, 1.4],
                [0, 0, 0],
                self.relativeChildPosition,
                self.relativeChildOrientation,
            )
        self.choice = choice = self.random.choice(2)
        self.goal = goals[choice]
        self.mission = missions[choice]

    def get_observation(
        self,
        cameraYaw,
        mission,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos, _ = self._p.getBasePositionAndOrientation(self.mass)
        (_, _, rgbaPixels, _, _,) = self._p.getCameraImage(
            self.image_width,
            self.image_height,
            viewMatrix=self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=pos,
                distance=CAMERA_DISTANCE,
                yaw=cameraYaw,
                pitch=CAMERA_PITCH,
                roll=0,
                upAxisIndex=2,
            ),
            shadow=0,
            flags=self._p.ER_NO_SEGMENTATION_MASK,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgbaPixels = rgbaPixels[..., :-1].astype(np.float32)
        obs = Observation(
            image=rgbaPixels,
            mission=self.tokens[mission],
        )
        obs = astuple(obs)
        assert self.observation_space.contains(obs)
        return obs

    def generator(self):
        i = dict(mission=self.mission)

        self._p.resetBasePositionAndOrientation(self.mass, [0, 0, 0.6], [0, 0, 0, 1])
        cameraYaw = self.cameraYaw
        action = yield self.get_observation(cameraYaw, self.mission)

        for global_step in range(self.max_episode_steps):
            a = ACTIONS[action].value

            cameraYaw += a.turn
            x, y, _, _ = self._p.getQuaternionFromEuler(
                [np.pi, 0, np.deg2rad(2 * cameraYaw) + np.pi]
            )
            x_shift = a.forward * x
            y_shift = a.forward * y
            x, y, *_ = self._p.getBasePositionAndOrientation(self.mass)[0]
            new_x = np.clip(x + x_shift, -self.env_bounds, self.env_bounds)
            new_y = np.clip(y + y_shift, -self.env_bounds, self.env_bounds)
            self._p.changeConstraint(self.mass_cid, [new_x, new_y, -0.1], maxForce=10)
            self._p.stepSimulation()

            s = self.get_observation(cameraYaw, self.mission)
            if ACTIONS[action].value.take_picture:
                PIL.Image.fromarray(np.uint8(Observation(*s).image)).show()
            t = ACTIONS[action].value.done
            if t:
                (*goal_poss, pos), _ = zip(
                    *[
                        self._p.getBasePositionAndOrientation(g)
                        for g in (*self.goals, self.mass)
                    ]
                )
                dists = [np.linalg.norm(np.array(pos) - np.array(g)) for g in goal_poss]
                r = float(np.argmin(dists) == self.choice)
            else:
                r = 0
            action = yield s, r, t, i

        s = self.get_observation(cameraYaw, self.mission)
        r = 0
        t = True
        yield s, r, t, i

    def step(self, action: Union[np.ndarray, int]):
        if isinstance(action, np.ndarray):
            action = action.item()
        s, r, t, i = self.iterator.send(action)
        # if t:
        #     for goal in i["goals"]:
        #         self._p.removeBody(goal)
        return s, r, t, i

    def reset(self):
        self.iterator = self.generator()
        return next(self.iterator)

    def render(self, mode="human"):

        if mode == "human":
            self.is_render = True
            return np.array([])
        if mode == "rgb_array":
            raise NotImplementedError

    def close(self):
        self._p.disconnect()

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed=seed)
        return [seed]


def main():
    path = Path(Path.home(), "downloads/dataset")
    model_ids = list(get_model_ids())
    env = Env(
        tokenizer=GPT2Tokenizer.from_pretrained("gpt2"),
        urdfs=list(get_urdfs(path, model_ids)),
        is_render=True,
        max_episode_steps=10000000,
    )
    env.render(mode="human")
    t = True
    r = None
    printed_mission = False

    cameraYaw = None
    action = Actions.NO_OP

    mapping = {
        p.B3G_RIGHT_ARROW: Actions.RIGHT,
        p.B3G_LEFT_ARROW: Actions.LEFT,
        p.B3G_UP_ARROW: Actions.FORWARD,
        p.B3G_DOWN_ARROW: Actions.BACKWARD,
        p.B3G_RETURN: Actions.PICTURE,
        p.B3G_SPACE: Actions.DONE,
    }

    while True:
        try:
            if t:
                cameraYaw = CAMERA_YAW
                env.reset()
                printed_mission = False
                if r is not None:
                    print("Reward:", r)
            spherePos, orn = p.getBasePositionAndOrientation(env.mass)

            cameraTargetPosition = spherePos
            p.resetDebugVisualizerCamera(
                CAMERA_DISTANCE, cameraYaw, CAMERA_PITCH, cameraTargetPosition
            )

            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if v & p.KEY_WAS_RELEASED and k in mapping:
                    action = Actions.NO_OP
            for k, v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    action = mapping.get(k, Actions.NO_OP)

            turn = action.value.turn
            action_index = ACTIONS.index(action)
            o, r, t, i = env.step(action_index)
            if not printed_mission:
                print(i["mission"])
                printed_mission = True

            if action == Actions.PICTURE:
                action = Actions.NO_OP
            cameraYaw += turn

        except KeyboardInterrupt:
            print("Received keyboard interrupt. Exiting.")
            return


if __name__ == "__main__":
    main()
