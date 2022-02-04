import os
import pkgutil
import re
import string
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, List, NamedTuple, Set, Tuple, TypeVar, Union

import gym.spaces as spaces
import gym.utils.seeding
import numpy as np
import PIL.Image
import pybullet as p
from art import text2art
from colors import color
from gym.spaces import Box, MultiDiscrete
from pybullet_utils import bullet_client
from tap import Tap
from tqdm import tqdm

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


class Action(NamedTuple):
    yaw: float = 0
    forward: float = 0
    pitch: float = 0
    done: bool = False
    take_picture: bool = False


class Actions(Enum):
    LEFT = Action(yaw=45)
    RIGHT = Action(yaw=-45)
    FORWARD = Action(forward=1)
    BACKWARD = Action(forward=-1)
    DONE = Action(done=True)


class DebugActions(Enum):
    UP = Action(pitch=45)
    DOWN = Action(pitch=-45)
    PICTURE = Action(take_picture=True)
    NO_OP = Action()


ACTIONS = [*Actions, *DebugActions]


@dataclass
class Observation(Generic[M, I]):
    mission: M
    image: I

    def to_obs(
        self,
        obs_space: Union["Observation", gym.spaces.Tuple] = None,
        check_obs: bool = True,
    ):
        obs = astuple(self)
        if obs_space is not None and check_obs:
            if isinstance(obs_space, Observation):
                obs_space = obs_space.to_space()
            if not obs_space.contains(obs):
                for s, o in zip(obs_space.spaces, obs):
                    if not s.contains(o):
                        breakpoint()
        return obs

    def to_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Tuple(astuple(self))


class String(gym.Space):
    def sample(self):
        return "".join(self.np_random.choice(string.ascii_letters) for _ in range(10))

    def contains(self, x):
        return isinstance(x, str)


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


@dataclass
class Env(gym.Env):
    env_bounds: float
    image_size: int
    max_episode_steps: int
    step_size: float
    steps_per_action: int
    urdfs: List[URDF]

    is_render: bool = False
    metadata = {
        "render.modes": ["human", "rgb_array", "ascii"],
        "video.frames_per_second": 60,
    }
    model_name: str = "gpt2"
    rank: int = 0
    random_seed: int = 0
    starting_camera_pitch: float = CAMERA_PITCH

    def __post_init__(self):
        self._camera_pitch = self.starting_camera_pitch
        self._camera_yaw = 0
        self.random = np.random.default_rng(self.random_seed)
        obs_spaces: Observation[MultiDiscrete, Box]
        self.obs_spaces = obs_spaces = Observation(
            mission=String(),
            image=Box(
                low=0,
                high=255,
                shape=[self.image_size, self.image_size, 3],
            ),
        )
        self.observation_space = obs_spaces.to_space()
        self.action_space = spaces.Discrete(len(Actions))

        self.names_to_urdfs = defaultdict(list)
        for urdf in self.urdfs:
            self.names_to_urdfs[urdf.name].append(urdf)

        self.objects = None
        self.names = None
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

        self._egl_plugin = None
        if sys.platform == "linux":
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                # noinspection PyUnresolvedReferences
                self._egl_plugin = self._p.loadPlugin(
                    egl.get_filename(), "_eglRendererPlugin"
                )
            else:
                self._egl_plugin = self._p.loadPlugin("eglRendererPlugin")
            print("EGL renderering enabled.")

        self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        sphereRadius = 0.02
        mass = 1
        visualShapeId = 2
        colSphereId = self._p.createVisualShape(
            self._p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 0, 0, 1]
        )
        self.mass = self._p.createMultiBody(mass, colSphereId, visualShapeId, [0, 0, 0])

        self.mass_cid = self._p.createConstraint(
            self.mass,
            -1,
            -1,
            -1,
            self._p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0, 1],
        )
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, False)

        # self._p.setGravity(0, 0, -10)
        halfExtents = [3 * self.env_bounds, 3 * self.env_bounds, 0.1]
        # floor_collision = self._p.createCollisionShape(
        #     self._p.GEOM_BOX, halfExtents=halfExtents
        # )
        floor_visual = self._p.createVisualShape(
            self._p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1, 1, 1, 1]
        )
        self.level_height = 4
        for i in range(len(self.names_to_urdfs) + 1):
            self._p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=floor_visual,
                basePosition=[0, 0, self.level_height * i - 1],
            )

        names = list(self.names_to_urdfs)
        self.random.shuffle(names)
        first, *rest = names
        self.pairs: List[Tuple[str, str]] = list(zip(names, [*rest, first]))
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.objects: List[Tuple[int, int]] = list(self.reset_objects())
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    @staticmethod
    def ascii_of_image(image: np.ndarray):
        def rows():
            for row in image:
                yield "".join([(color("$$", tuple(rgb.astype(int)))) for rgb in row])

        return "\n".join(rows())

    def close(self):
        self._p.disconnect()

    def generator(self):
        goal = self.random.choice(2)
        level = self.random.choice(len(self.names_to_urdfs))
        names = self.pairs[level]
        mission = names[goal]
        objects = self.objects[level]

        i = dict(mission=mission)

        z = 1
        self._camera_yaw = self.random.choice(360)
        self._camera_pitch = self.starting_camera_pitch
        mass_start_pos = self.random.uniform(low=-np.ones(3), high=np.ones(3))
        z += self.level_height * level
        mass_start_pos[-1] = z

        self._p.resetBasePositionAndOrientation(self.mass, mass_start_pos, [0, 0, 0, 1])
        action = yield self.get_observation(
            camera_yaw=self._camera_yaw,
            camera_pitch=self._camera_pitch,
            mission=mission,
        )

        for global_step in range(self.max_episode_steps):

            self._camera_yaw += action.yaw
            self._camera_pitch += action.pitch

            x, y, _, _ = self._p.getQuaternionFromEuler(
                [np.pi, 0, np.deg2rad(2 * self._camera_yaw) + np.pi]
            )
            x_shift = self.step_size * action.forward * x
            y_shift = self.step_size * action.forward * y
            x, y, *_ = self._p.getBasePositionAndOrientation(self.mass)[0]
            new_x = np.clip(x + x_shift, -self.env_bounds, self.env_bounds)
            new_y = np.clip(y + y_shift, -self.env_bounds, self.env_bounds)
            self._p.changeConstraint(self.mass_cid, [new_x, new_y, z], maxForce=100)
            for _ in range(self.steps_per_action):
                self._p.stepSimulation()

            s = self.get_observation(
                camera_yaw=self._camera_yaw,
                camera_pitch=self._camera_pitch,
                mission=mission,
            )
            if action.take_picture:
                PIL.Image.fromarray(self.render(mode="rgb_array")).show()
            if action.done:
                (*goal_poss, pos), _ = zip(
                    *[
                        self._p.getBasePositionAndOrientation(g)
                        for g in (*objects, self.mass)
                    ]
                )
                dists = [np.linalg.norm(np.array(pos) - np.array(g)) for g in goal_poss]
                r = float(np.argmin(dists) == goal)
            else:
                r = 0
            self._r = r
            action = yield s, r, action.done, i

        s = self.get_observation(
            camera_yaw=self._camera_yaw,
            camera_pitch=self._camera_pitch,
            mission=mission,
        )
        r = 0
        t = True
        yield s, r, t, i

    def reset_objects(self):

        for i, names in enumerate(tqdm(self.pairs, position=self.rank)):
            objects: List[int] = []
            for base_position, name in zip(
                [
                    [-self.env_bounds / 1, self.env_bounds / 1, self.level_height * i],
                    [self.env_bounds / 1, -self.env_bounds / 1, self.level_height * i],
                ],
                names,
            ):
                urdfs = self.names_to_urdfs[name]
                urdf = urdfs[self.random.choice(len(urdfs))]

                try:
                    goal: int = self._p.loadURDF(
                        str(urdf.path),
                        basePosition=base_position,
                        globalScaling=10,
                        useFixedBase=True,
                    )
                except self._p.error:
                    print(self._p.error)
                    raise RuntimeError(f"Error while loading {urdf.path}")

                collisionFilterGroup = 0x2
                collisionFilterMask = 0x2
                for j in range(p.getNumJoints(goal)):
                    p.setCollisionFilterGroupMask(
                        goal, j, collisionFilterGroup, collisionFilterMask
                    )

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
                    [0, 0, 0],
                    [0, 0, 0, 1],
                )
            o1, o2 = objects
            yield o1, o2

    def get_observation(
        self,
        camera_yaw: float,
        camera_pitch: float,
        mission: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos, _ = self._p.getBasePositionAndOrientation(self.mass)

        viewMatrix = self._p.computeViewMatrixFromYawPitchRoll(
            pos, CAMERA_DISTANCE, camera_yaw, camera_pitch, ROLL, UP_AXIS_INDEX
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
        return obs.to_obs(self.observation_space)

    def render(self, mode="human", pause=True):
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

    def reset(self):
        self.iterator = self.generator()
        return next(self.iterator)

    def step(self, action: Union[Actions, DebugActions, np.ndarray, int]):
        if isinstance(action, np.ndarray):
            action = action.item()
        if isinstance(action, int):
            self._a = ACTIONS[action]
            action = self._a.value
        assert isinstance(action, Action)
        return self.iterator.send(action)


class Args(Tap):
    data_path: str = str(
        Path(Path.home(), ".cache/data/pybullet-URDF-models/urdf_models/models")
    )
    env_bounds: float = 1
    image_size: int = 84
    max_episode_steps: int = 200
    step_size: float = 1
    steps_per_action: int = 4


class DebugArgs(Args):
    mode: str = "ascii"
    names: str = None
    urdf_index: int = 0


def main(args: DebugArgs):
    path = Path(args.data_path)
    if args.names is None:
        names = args.names
    else:
        names = set(args.names.split(","))
    urdfs = list(get_urdfs(path, names))
    env = Env(
        env_bounds=args.env_bounds,
        image_size=args.image_size,
        is_render=args.mode == "human",
        max_episode_steps=args.max_episode_steps,
        step_size=args.step_size,
        steps_per_action=args.steps_per_action,
        urdfs=urdfs,
    )

    def render():
        env.render(mode=args.mode, pause=False)

    t = True
    r = None

    if args.mode == "human":
        render()
        mapping = {
            p.B3G_RIGHT_ARROW: Actions.RIGHT,
            p.B3G_LEFT_ARROW: Actions.LEFT,
            p.B3G_UP_ARROW: Actions.FORWARD,
            p.B3G_DOWN_ARROW: Actions.BACKWARD,
            p.B3G_PAGE_UP: DebugActions.UP,
            p.B3G_PAGE_DOWN: DebugActions.DOWN,
            p.B3G_RETURN: DebugActions.PICTURE,
            p.B3G_SPACE: Actions.DONE,
        }
    elif args.mode == "ascii":
        mapping = {
            "d": Actions.RIGHT,
            "a": Actions.LEFT,
            "w": Actions.FORWARD,
            "s": Actions.BACKWARD,
            "e": DebugActions.UP,
            "c": DebugActions.DOWN,
            "p": DebugActions.PICTURE,
            "x": Actions.DONE,
        }
    else:
        raise RuntimeError()

    while True:
        try:
            if t:
                o = env.reset()
                if args.mode == "human":
                    if r is not None:
                        print("Reward:", r)
                    print(Observation(*o).mission)

            render()

            action = None

            if args.mode == "human":
                keys = p.getKeyboardEvents()
                for k, v in keys.items():
                    if v & p.KEY_WAS_RELEASED and k in mapping:
                        action = mapping.get(k, None)
                if action is None:
                    action = DebugActions.NO_OP
                else:
                    print("Action:", action)

            elif args.mode == "ascii":
                while action is None:
                    key = input("enter action key:")
                    if key in mapping:
                        action = mapping[key]

            action_index = ACTIONS.index(action)
            o, r, t, i = env.step(action_index)

            if args.mode == "human":
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Received keyboard interrupt. Exiting.")
            return


if __name__ == "__main__":
    main(DebugArgs().parse_args())
