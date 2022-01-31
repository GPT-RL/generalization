import json
import os
import string
import sys
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, NamedTuple, Set, Tuple, TypeVar, Union

import gym.spaces as spaces
import gym.utils.seeding
import numpy as np
import PIL.Image
import pybullet as p
import yaml
from gym.spaces import Box, MultiDiscrete
from pybullet_utils import bullet_client

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


def get_urdfs(path: Path, names: Set[str] = None):
    with Path("models.yml").open() as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
    for subdir in path.iterdir():
        with Path(subdir, "meta.json").open() as f:
            meta = json.load(f)
        name = meta["model_cat"]
        if names is not None and name not in names:
            continue
        if subdir.name not in models:
            continue
        urdf = Path(subdir, "mobility.urdf")
        assert urdf.exists()
        with Path(subdir, "bounding_box.json").open() as f:
            box = json.load(f)
        _, _, z_min = box["min"]
        yield URDF(name=name, path=urdf, z=-z_min)


class String(gym.Space):
    def sample(self):
        return "".join(self.np_random.choice(string.ascii_letters) for _ in range(10))

    def contains(self, x):
        return isinstance(x, str)


@dataclass
class Env(gym.Env):
    urdfs: Tuple[URDF, URDF]
    camera_yaw: float = CAMERA_YAW
    env_bounds: float = 5
    image_height: float = 64  # 72
    image_width: float = 64  # 96
    is_render: bool = False
    max_episode_steps: int = 200
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}
    model_name: str = "gpt2"
    random_seed: int = 0
    reindex_tokens: bool = False

    def __post_init__(self):
        self._camera_yaw = self.camera_yaw
        names = [urdf.name for urdf in self.urdfs]
        assert len(set(names)) == 2, names
        self.random = np.random.default_rng(self.random_seed)
        obs_spaces: Observation[MultiDiscrete, Box] = Observation(
            mission=String(),
            image=Box(
                low=0,
                high=255,
                shape=[self.image_height, self.image_width, 3],
            ),
        )
        self.observation_space = spaces.Tuple(astuple(obs_spaces))
        self.action_space = spaces.Discrete(5)

        self.iterator = None

        # initialize simulator
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

        relativeChildPosition = [0, 0, 0]
        relativeChildOrientation = [0, 0, 0, 1]

        self.mass_cid = self._p.createConstraint(
            self.mass,
            -1,
            -1,
            -1,
            self._p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            relativeChildPosition,
            relativeChildOrientation,
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

        self.choice = choice = self.random.choice(2)
        self.mission = names[choice]
        self.objects = objects = []

        print()
        for u in self.urdfs:
            print(u.name, u.path.parent.name)
        for base_position, urdf in zip(
            [
                [self.env_bounds / 3, self.env_bounds / 3, 0],
                [-self.env_bounds / 3, -self.env_bounds / 3, 0],
            ],
            self.urdfs,
        ):
            base_position[-1] = urdf.z

            try:
                with suppress_stdout():
                    goal = self._p.loadURDF(
                        str(urdf.path), basePosition=base_position, useFixedBase=True
                    )
            except self._p.error:
                print(self._p.error)
                raise RuntimeError(f"Error while loading {urdf.path}")
            objects.append(goal)

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
                relativeChildPosition,
                relativeChildOrientation,
            )

    def get_observation(
        self,
        camera_yaw,
        mission,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos, _ = self._p.getBasePositionAndOrientation(self.mass)
        (_, _, rgbaPixels, _, _,) = self._p.getCameraImage(
            self.image_width,
            self.image_height,
            viewMatrix=self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=pos,
                distance=CAMERA_DISTANCE,
                yaw=camera_yaw,
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
            mission=mission,
        )
        obs = astuple(obs)
        assert self.observation_space.contains(obs)
        return obs

    def generator(self):
        i = dict(mission=self.mission)

        self._p.resetBasePositionAndOrientation(self.mass, [0, 0, 0.6], [0, 0, 0, 1])
        self._camera_yaw = self.camera_yaw
        action = yield self.get_observation(self._camera_yaw, self.mission)

        for global_step in range(self.max_episode_steps):
            a = ACTIONS[action].value

            self._camera_yaw += a.turn
            x, y, _, _ = self._p.getQuaternionFromEuler(
                [np.pi, 0, np.deg2rad(2 * self._camera_yaw) + np.pi]
            )
            x_shift = a.forward * x
            y_shift = a.forward * y
            x, y, *_ = self._p.getBasePositionAndOrientation(self.mass)[0]
            new_x = np.clip(x + x_shift, -self.env_bounds, self.env_bounds)
            new_y = np.clip(y + y_shift, -self.env_bounds, self.env_bounds)
            self._p.changeConstraint(self.mass_cid, [new_x, new_y, -0.1], maxForce=10)
            self._p.stepSimulation()

            s = self.get_observation(self._camera_yaw, self.mission)
            if ACTIONS[action].value.take_picture:
                PIL.Image.fromarray(np.uint8(Observation(*s).image)).show()
            t = ACTIONS[action].value.done
            if t:
                (*goal_poss, pos), _ = zip(
                    *[
                        self._p.getBasePositionAndOrientation(g)
                        for g in (*self.objects, self.mass)
                    ]
                )
                dists = [np.linalg.norm(np.array(pos) - np.array(g)) for g in goal_poss]
                r = float(np.argmin(dists) == self.choice)
            else:
                r = 0
            action = yield s, r, t, i

        s = self.get_observation(self._camera_yaw, self.mission)
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

            cameraTargetPosition, orn = p.getBasePositionAndOrientation(self.mass)

            self._p.resetDebugVisualizerCamera(
                CAMERA_DISTANCE, self._camera_yaw, CAMERA_PITCH, cameraTargetPosition
            )
            return np.array([])
        if mode == "rgb_array":
            raise NotImplementedError

    def close(self):
        self._p.disconnect()


def main():
    path = Path(Path.home(), "downloads/dataset")
    urdf1, urdf2, *_ = get_urdfs(path)
    env = Env(
        urdfs=(urdf1, urdf2),
        is_render=True,
        max_episode_steps=10000000,
    )
    env.render(mode="human")
    t = True
    r = None
    printed_mission = False

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
                env.reset()
                printed_mission = False
                if r is not None:
                    print("Reward:", r)

            env.render()

            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if v & p.KEY_WAS_RELEASED and k in mapping:
                    action = Actions.NO_OP
            for k, v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    action = mapping.get(k, Actions.NO_OP)

            action_index = ACTIONS.index(action)
            o, r, t, i = env.step(action_index)
            if not printed_mission:
                print(Observation(*o).mission)
                printed_mission = True

            if action == Actions.PICTURE:
                action = Actions.NO_OP

        except KeyboardInterrupt:
            print("Received keyboard interrupt. Exiting.")
            return


if __name__ == "__main__":
    main()
