import os
import re
import string
import sys
import time
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
import pybullet_data
from art import text2art
from colors import color
from gym.spaces import Box, MultiDiscrete
from pybullet_utils import bullet_client
from tap import Tap

CAMERA_DISTANCE = 0.2
CAMERA_PITCH = -10
CAMERA_YAW = 315

ROLL = 0
UP_AXIS_INDEX = 2
NEAR_PLANE = 0.01
FAR_PLANE = 100

FOV = 60

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
    LEFT = Action(45, 0)
    RIGHT = Action(-45, 0)
    FORWARD = Action(0, 1)
    BACKWARD = Action(0, -1)
    DONE = Action(done=True)


class DebugActions(Enum):
    PICTURE = Action(take_picture=True)
    NO_OP = Action()


ACTIONS = [*Actions, *DebugActions]


class URDF(NamedTuple):
    name: str
    path: Path
    # z: float


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
    for subdir in path.iterdir():
        path = Path(subdir, "model.urdf")
        if path.exists():
            name = subdir.name.replace("_", " ")
            name = re.sub(r" \d", "", name)
            if names is None or name in names:
                yield URDF(name=name, path=path)  # , z=-z_min)


class String(gym.Space):
    def sample(self):
        return "".join(self.np_random.choice(string.ascii_letters) for _ in range(10))

    def contains(self, x):
        return isinstance(x, str)


@dataclass
class Env(gym.Env):
    image_size: float
    max_episode_steps: int
    steps_per_action: int
    urdfs: Tuple[URDF, URDF]
    camera_yaw: float = CAMERA_YAW
    env_bounds: float = 1
    is_render: bool = False
    metadata = {
        "render.modes": ["human", "rgb_array", "ascii"],
        "video.frames_per_second": 60,
    }
    model_name: str = "gpt2"
    rank: int = 0
    random_seed: int = 0

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
                shape=[self.image_size, self.image_size, 3],
            ),
        )
        self.observation_space = spaces.Tuple(astuple(obs_spaces))
        self.action_space = spaces.Discrete(len(Actions))

        self.iterator = None
        self._s = None
        self._a = None
        self._r = None

        # initialize simulator

        if self.is_render:
            with suppress_stdout():
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)
        else:
            self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        sphereRadius = 0.02
        mass = 1
        visualShapeId = 2
        colSphereId = self._p.createVisualShape(
            self._p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 0, 0, 1]
        )
        self.mass = self._p.createMultiBody(mass, colSphereId, visualShapeId, [0, 0, 0])

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
        # floor_collision = self._p.createCollisionShape(
        #     self._p.GEOM_BOX, halfExtents=halfExtents
        # )
        floor_visual = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1, 1, 1, 0.5]
        )
        self._p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=floor_visual, basePosition=[0, 0, -1]
        )

        self.names = names
        self.objects = objects = []

        print()
        print(
            f"Env {self.rank}: {', '.join([u.name for u in self.urdfs])}",
        )
        for base_position, urdf in zip(
            [
                [-self.env_bounds / 1, self.env_bounds / 1, 0],
                [self.env_bounds / 1, -self.env_bounds / 1, 0],
            ],
            self.urdfs,
        ):

            try:
                with suppress_stdout():
                    goal = self._p.loadURDF(
                        str(urdf.path),
                        basePosition=base_position,
                        globalScaling=10,
                        useFixedBase=True,
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

        viewMatrix = self._p.computeViewMatrixFromYawPitchRoll(
            pos, CAMERA_DISTANCE, camera_yaw, CAMERA_PITCH, ROLL, UP_AXIS_INDEX
        )
        aspect = 1
        projectionMatrix = self._p.computeProjectionMatrixFOV(
            FOV, aspect, NEAR_PLANE, FAR_PLANE
        )
        (_, _, rgbaPixels, _, _,) = self._p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix,
            projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgbaPixels = rgbaPixels[..., :-1].astype(np.float32)
        obs = Observation(
            image=rgbaPixels,
            mission=mission,
        )
        self._s = obs
        obs = astuple(obs)
        assert self.observation_space.contains(obs)
        return obs

    def generator(self):

        goal = self.random.choice(2)
        mission = self.names[goal]

        i = dict(mission=mission)

        z = 1
        self._camera_yaw = self.random.choice(360)
        mass_start_pos = self.random.uniform(low=-np.ones(3), high=np.ones(3))
        mass_start_pos[-1] = z

        self._p.resetBasePositionAndOrientation(self.mass, mass_start_pos, [0, 0, 0, 1])
        action = yield self.get_observation(self._camera_yaw, mission)

        for global_step in range(self.max_episode_steps):

            self._camera_yaw += action.turn

            x, y, _, _ = self._p.getQuaternionFromEuler(
                [np.pi, 0, np.deg2rad(2 * self._camera_yaw) + np.pi]
            )
            x_shift = action.forward * x
            y_shift = action.forward * y
            x, y, *_ = self._p.getBasePositionAndOrientation(self.mass)[0]
            new_x = np.clip(x + x_shift, -self.env_bounds, self.env_bounds)
            new_y = np.clip(y + y_shift, -self.env_bounds, self.env_bounds)
            self._p.changeConstraint(self.mass_cid, [new_x, new_y, z], maxForce=100)
            for _ in range(self.steps_per_action):
                self._p.stepSimulation()

            s = self.get_observation(self._camera_yaw, mission)
            if action.take_picture:
                PIL.Image.fromarray(self.render(mode="rgb_array")).show()
            if action.done:
                (*goal_poss, pos), _ = zip(
                    *[
                        self._p.getBasePositionAndOrientation(g)
                        for g in (*self.objects, self.mass)
                    ]
                )
                dists = [np.linalg.norm(np.array(pos) - np.array(g)) for g in goal_poss]
                r = float(np.argmin(dists) == goal)
            else:
                r = 0
            self._r = r
            action = yield s, r, action.done, i

        s = self.get_observation(self._camera_yaw, mission)
        r = 0
        t = True
        yield s, r, t, i

    def step(self, action: Union[Actions, DebugActions, np.ndarray, int]):
        if isinstance(action, np.ndarray):
            action = action.item()
        if isinstance(action, int):
            self._a = ACTIONS[action]
            action = self._a.value
        assert isinstance(action, Action)
        return self.iterator.send(action)

    def reset(self):
        self.iterator = self.generator()
        return next(self.iterator)

    @staticmethod
    def ascii_of_image(image: np.ndarray):
        def rows():
            for row in image:
                yield "".join([(color("$$", tuple(rgb.astype(int)))) for rgb in row])

        return "\n".join(rows())

    def render(self, mode="human", pause=True):
        if mode == "human":
            self.is_render = True

            cameraTargetPosition, orn = p.getBasePositionAndOrientation(self.mass)

            self._p.resetDebugVisualizerCamera(
                CAMERA_DISTANCE, self._camera_yaw, CAMERA_PITCH, cameraTargetPosition
            )
            return np.array([])
        if mode == "rgb_array":
            return np.uint8(self._s.image)
        if mode == "ascii":
            print(self.ascii_of_image(self._s.image))
            print()
            subtitle = self._s.mission.upper()
            if self._a is not None:
                subtitle = f"{subtitle}, {self._a.name}, r={self._r}"
            print(text2art(subtitle, font="com_sen"))
            if pause:
                input("Press enter to continue.")

    def close(self):
        self._p.disconnect()


class Args(Tap):
    data_path: str = str(
        Path(Path.home(), ".cache/data/pybullet-URDF-models/urdf_models/models")
    )


def main(args: Args):
    path = Path(args.data_path)
    urdf1, urdf2, *_ = get_urdfs(path)
    env = Env(
        urdfs=(urdf1, urdf2),
        image_size=64,
        is_render=False,
        max_episode_steps=10000000,
        steps_per_action=100,
    )
    env.render(mode="human")
    t = True
    r = None

    mapping = {
        "d": Actions.RIGHT,
        "a": Actions.LEFT,
        "w": Actions.FORWARD,
        "s": Actions.BACKWARD,
        "p": DebugActions.PICTURE,
        "x": Actions.DONE,
    }

    while True:
        try:
            if t:
                env.reset()
                if r is not None:
                    print("Reward:", r)

            env.render("ascii", pause=False)
            action = None
            while action is None:
                key = input("enter action key:")
                if key in mapping:
                    action = mapping[key]

            action_index = ACTIONS.index(action)
            o, r, t, i = env.step(action_index)

            time.sleep(0.05)

        except KeyboardInterrupt:
            print("Received keyboard interrupt. Exiting.")
            return


if __name__ == "__main__":
    main(Args().parse_args())
