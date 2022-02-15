import itertools
import math
import string
import typing
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, TypeVar, Union, cast

import gym
import numpy as np
from art import text2art
from colors import color
from gym import Space, spaces
from gym_miniworld.entity import Entity
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.params import DEFAULT_PARAMS
from my.entities import Box, MeshEnt
from tap import Tap

EXCLUDED = "excluded"
DESCRIPTION = "description"
NAME = "name"
PAIR = "pair"
PATH = "path"


class Args(Tap):
    data_path: str = Path(Path.home(), ".cache/data/ycb")
    floor_tex: str = "floor_tiles_white"
    image_size: int = 128
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


RADIUS = 1.5724637533369341


class Env(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(
        self,
        floor_tex: str,
        image_size: int,
        meshes: List[Mesh],
        room_size: float,
        max_episode_steps: int = 180,
        pitch: float = -30,
        rank: int = 0,
        **kwargs,
    ):
        self.floor_tex = floor_tex
        self.rank = rank
        assert room_size >= 2
        self.size = room_size

        self.meshes = OrderedDict(
            [(m.name, m) for m in sorted(meshes, key=lambda m: m.name)]
        )

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
            floor_tex=self.floor_tex,
        )
        while True:
            if self._mesh_names is None:
                self._mission, self._dist_name = mesh_names = self.rand.subset(
                    list(self.meshes), num_elems=2
                )
            else:
                mesh_names = self._mission, self._dist_name = self._mesh_names
            meshes = [self.meshes[n] for n in mesh_names]

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
                for ent in self.entities:
                    ent.radius = RADIUS
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

    def can_pick_up(self, ent: Entity):
        test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
        entities = self.entities
        self.entities = [ent]
        intersects = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
        self.entities = entities
        return intersects is ent

    def generator(self):
        action = None
        reward = None
        done = False
        highlighted = {}

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
                info.update({PAIR: (self._mission, self._dist_name)})
            action = yield obs, reward, done, info
            action = cast(MiniWorldEnv.Actions, action)
            assert not done

            image, reward, done, info = super().step(action)
            for ent in [self.goal, self.dist]:
                if self.can_pick_up(ent):
                    box = Box(np.array([1, 1, 1]))
                    if ent not in highlighted:
                        highlighted[ent] = box
                        self.place_entity(ent=box, pos=ent.pos, dir=ent.dir)
                elif ent in highlighted:
                    self.entities.remove(highlighted[ent])
                    del highlighted[ent]

            if action == self.actions.done:
                done = True
            if self.agent.carrying == self.goal:
                reward += self._reward()
                done = True
            elif self.agent.carrying == self.dist:
                done = True

    def make_obs(self, image: np.ndarray) -> Obs:
        return Obs(image=image, mission=self._mission)

    def move_agent(self, fwd_dist, fwd_drift):
        """
        Move the agent forward
        """

        next_pos = (
            self.agent.pos
            + self.agent.dir_vec * fwd_dist
            + self.agent.right_vec * fwd_drift
        )

        # if self.intersect(self.agent, next_pos, self.agent.radius):
        #     return False

        carrying = self.agent.carrying
        if carrying:
            next_carrying_pos = self._get_carry_pos(next_pos, carrying)

            if self.intersect(carrying, next_carrying_pos, carrying.radius):
                return False

            carrying.pos = next_carrying_pos

        self.agent.pos = next_pos

        return True

    def render(self, mode="human", pause=True, **kwargs):
        if mode == "ascii":
            self._render(pause=pause)
        else:
            return super().render(mode=mode, **kwargs)

    def reset(self, mesh_names: typing.Tuple[str, str] = None) -> dict:
        self._mesh_names = mesh_names
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
