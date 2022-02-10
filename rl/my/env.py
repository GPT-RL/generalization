import itertools
import math
import pickle
import string
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, TypeVar, Union, cast

import gym
import numpy as np
import redis
from art import text2art
from colors import color
from gym import Space, spaces
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.params import DEFAULT_PARAMS
from my.mesh_ent import MeshEnt
from tap import Tap

EXCLUDED = "excluded"
DESCRIPTION = "description"
NAME = "name"
PATH = "path"


class Args(Tap):
    data_path: str = Path(Path.home(), ".cache/data/ycb")
    names: Optional[str] = None
    room_size: float = 8
    obj_pattern: str = "*/*/*.obj"
    png_pattern: str = "*/*/*.png"
    seed: int = 0


class Mesh(NamedTuple):
    obj: Union[Path, str]
    png: Optional[Path]
    name: str
    height: float = 1


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


R = redis.Redis()


class Env(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(
        self,
        meshes: List[Mesh],
        size: float,
        image_size: int = 128,
        max_episode_steps: int = 180,
        pitch: float = -30,
        rank: int = 0,
        **kwargs,
    ):
        self.rank = rank
        assert size >= 2
        self.size = size
        self.timestep = 0

        self.meshes = sorted(meshes, key=lambda m: m.name)

        params = deepcopy(DEFAULT_PARAMS)
        params.set("cam_pitch", pitch, pitch, pitch)
        self._iterator = None
        self._render = None
        self._mission = None
        self._dist_name = None
        super().__init__(
            max_episode_steps=max_episode_steps,
            params=params,
            obs_width=image_size,
            obs_height=image_size,
            **kwargs,
        )
        # Allow only movement actions (left/right/forward) and pickup
        self.action_space = spaces.Discrete(self.actions.pickup + 1)
        self.observation_space = Obs(
            image=self.observation_space, mission=String()
        ).to_space()

    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            floor_tex="floor_tiles_white",
        )
        while True:
            meshes = self.rand.subset(self.meshes, num_elems=2)
            self._mission, self._dist_name = [m.name for m in meshes]

            try:
                self.goal, self.dist = [
                    self.place_entity(
                        MeshEnt(
                            str(mesh.obj),
                            height=mesh.height,
                            static=False,
                            tex_name=str(mesh.png) if mesh.png else None,
                        ),
                        name=mesh.name,
                    )
                    for mesh in meshes
                ]
                break
            except PlacementError:
                print("Failed to place:", self._mission, self._dist_name)
                continue

        self.place_agent()

    @staticmethod
    def ascii_of_image(image: np.ndarray):
        def rows():
            for row in image:
                yield "".join([(color("██", tuple(rgb.astype(int)))) for rgb in row])

        return "\n".join(rows())

    def generator(self):
        action = None
        reward = None
        done = False

        def render(pause=True):
            print(self.ascii_of_image(self.render_obs()))
            print()
            subtitle = self._mission
            if action is not None:
                subtitle += f", {action.name.replace('_', ' ')}"
            if reward is not None:
                subtitle += f", r={round(reward, 2)}"
            if done:
                subtitle += ", done"
            print(text2art(subtitle.swapcase(), font="com_sen"))
            if pause:
                input("Press enter to continue.")

        self._render = render

        image = super().reset()
        info = {}
        while True:
            obs = self.make_obs(image)
            if done:
                info.update(pair=(self._mission, self._dist_name))

            R.set(f"{self.timestep},{self.rank}", pickle.dumps((obs, action)))
            self.timestep += 1
            action = yield obs, reward, done, info
            action = cast(MiniWorldEnv.Actions, action)
            assert not done

            image, reward, done, info = super().step(action)

            if action == self.actions.done:
                done = True
            if self.agent.carrying == self.goal:
                reward += self._reward()
                done = True
            elif self.agent.carrying == self.dist:
                done = True

    def make_obs(self, image: np.ndarray) -> Obs:
        return Obs(image=image, mission=self._mission)

    def render(self, mode="human", pause=True, **kwargs):
        if mode == "ascii":
            self._render(pause=pause)
        else:
            return super().render(mode=mode, **kwargs)

    def reset(self) -> dict:
        self._iterator = self.generator()
        obs, _, _, _ = next(self._iterator)
        return self.to_obs(obs)

    def step(self, action: Union[np.ndarray, MiniWorldEnv.Actions]):
        action: MiniWorldEnv.Actions = (
            action
            if isinstance(action, MiniWorldEnv.Actions)
            else [*MiniWorldEnv.Actions][action.item()]
        )
        obs, reward, done, info = self._iterator.send(action)
        return self.to_obs(obs), reward, done, info

    def to_obs(self, obs: Obs):
        # check = isinstance(self.observation_space, gym.spaces.Dict)
        return obs.to_obs(self.observation_space)

    def place_entity(
        self,
        ent,
        room=None,
        pos=None,
        dir=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None,
        name=None,
    ):
        """
        Place an entity/object in the world.
        Find a position that doesn't intersect with any other object.
        """

        assert len(self.rooms) > 0, "create rooms before calling place_entity"
        assert ent.radius is not None, "entity must have physical size defined"

        # Generate collision detection data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # If an exact position if specified
        if pos is not None:
            ent.dir = dir if dir is not None else self.rand.float(-math.pi, math.pi)
            ent.pos = pos
            self.entities.append(ent)
            return ent

        # Keep retrying until we find a suitable position
        for i in itertools.count():
            # Pick a room, sample rooms proportionally to floor surface area
            r = room if room else self.rand.choice(self.rooms, probs=self.room_probs)

            # Choose a random point within the square bounding box of the room
            lx = r.min_x if min_x is None else min_x
            hx = r.max_x if max_x is None else max_x
            lz = r.min_z if min_z is None else min_z
            hz = r.max_z if max_z is None else max_z
            pos = self.rand.float(
                low=[lx + ent.radius, 0, lz + ent.radius],
                high=[hx - ent.radius, 0, hz - ent.radius],
            )

            # Make sure the position is within the room's outline
            if not r.point_inside(pos):
                if i > 1000:
                    raise PlacementError()

                continue

            # Make sure the position doesn't intersect with any walls
            if self.intersect(ent, pos, ent.radius):
                if i > 1000:
                    raise PlacementError()
                continue

            # Pick a direction
            d = dir if dir is not None else self.rand.float(-math.pi, math.pi)

            ent.pos = pos
            ent.dir = d
            break

        self.entities.append(ent)

        return ent
