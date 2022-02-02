import os
import re
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
from gym.spaces import Box, MultiDiscrete
from pybullet_utils import bullet_client

CAMERA_DISTANCE = 0.1
CAMERA_PITCH = -40
CAMERA_YAW = 315

M = TypeVar("M")
I = TypeVar("I")


@dataclass
class Observation(Generic[M, I]):
    mission: M
    image: I


class Action(NamedTuple):
    turn: float = 0
    forward: float = 0
    pitch: float = 0
    done: bool = False
    take_picture: bool = False


class Actions(Enum):
    LEFT = Action(3, 0, 0)
    RIGHT = Action(-3, 0, 0)
    FORWARD = Action(0, 0.18, 0)
    BACKWARD = Action(0, -0.18, 0)
    UP = Action(0, 0, 3)
    DOWN = Action(0, 0, -3)
    DONE = Action(done=True)
    PICTURE = Action(take_picture=True)
    NO_OP = Action()


ACTIONS = [*Actions]


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
    camera_pitch: float = CAMERA_PITCH
    camera_yaw: float = CAMERA_YAW
    env_bounds: float = 0.4
    is_render: bool = False
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}
    model_name: str = "gpt2"
    rank: int = 0
    random_seed: int = 0

    def __post_init__(self):
        self._camera_yaw = self.camera_yaw
        self._camera_pitch = self.camera_pitch
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
        self.action_space = spaces.Discrete(
            len(Actions) - 2
        )  # -2 to prohibit no_op and camera

        self.iterator = None

        # initialize simulator

        if self.is_render:
            with suppress_stdout():
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)
        else:
            with suppress_stdout():
                self._p = bullet_client.BulletClient(connection_mode=p.DIRECT)

        sphereRadius = 0.02
        mass = 1
        visualShapeId = 2
        colSphereId = self._p.createVisualShape(
            self._p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 0, 0, 1]
        )
        self.mass_start_pos = [0, 0, 0]
        self.mass = self._p.createMultiBody(
            mass, colSphereId, visualShapeId, self.mass_start_pos
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
        # floor_collision = self._p.createCollisionShape(
        #     self._p.GEOM_BOX, halfExtents=halfExtents
        # )
        floor_visual = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1, 1, 1, 0.2]
        )
        self._p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=floor_visual, basePosition=[0, 0, -0.4]
        )

        self.choice = choice = self.random.choice(2)
        self.mission = names[choice]
        self.objects = objects = []

        print()
        print(
            f"Env {self.rank}: {', '.join([u.name for u in self.urdfs])}",
        )
        for base_position, urdf in zip(
            [
                [-self.env_bounds / 3, self.env_bounds / 3, 0],
                [self.env_bounds / 3, -self.env_bounds / 3, 0],
            ],
            self.urdfs,
        ):
            base_position[-1] = -0.2  # urdf.z

            try:
                with suppress_stdout():
                    goal = self._p.loadURDF(
                        str(urdf.path),
                        basePosition=base_position,
                        globalScaling=2,
                        useFixedBase=True,
                    )
            except self._p.error:
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
            self.image_size,
            self.image_size,
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

        self._p.resetBasePositionAndOrientation(
            self.mass, self.mass_start_pos, [0, 0, 0, 1]
        )
        self._camera_yaw = self.camera_yaw
        self._camera_pitch = self.camera_pitch
        action = yield self.get_observation(self._camera_yaw, self.mission)

        for global_step in range(self.max_episode_steps):
            a = ACTIONS[action].value

            self._camera_yaw += a.turn
            self._camera_pitch += a.pitch
            self._camera_pitch = int(np.clip(self._camera_pitch, -45, 45 / 2))

            x, y, _, _ = self._p.getQuaternionFromEuler(
                [np.pi, 0, np.deg2rad(2 * self._camera_yaw) + np.pi]
            )
            x_shift = a.forward * x
            y_shift = a.forward * y
            x, y, *_ = self._p.getBasePositionAndOrientation(self.mass)[0]
            new_x = np.clip(x + x_shift, -self.env_bounds, self.env_bounds)
            new_y = np.clip(y + y_shift, -self.env_bounds, self.env_bounds)
            self._p.changeConstraint(self.mass_cid, [new_x, new_y, -0.1], maxForce=100)
            for _ in range(self.steps_per_action):
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
                CAMERA_DISTANCE,
                self._camera_yaw,
                self._camera_pitch,
                cameraTargetPosition,
            )
            return np.array([])
        if mode == "rgb_array":
            raise NotImplementedError

    def close(self):
        self._p.disconnect()


def main():
    path = Path(Path.home(), "pybullet-URDF-models/urdf_models/models")
    urdf1, urdf2, *_ = get_urdfs(path)
    env = Env(
        urdfs=(urdf1, urdf2),
        image_size=64,
        is_render=True,
        max_episode_steps=10000000,
        steps_per_action=1,
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
        p.B3G_PAGE_UP: Actions.UP,
        p.B3G_PAGE_DOWN: Actions.DOWN,
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
