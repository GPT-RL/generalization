import string
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, NamedTuple, Optional, TypeVar, Union

import gym
import numpy as np
from art import text2art
from colors import color
from gym import Space, spaces
from gym_miniworld.envs import OneRoom
from gym_miniworld.miniworld import MiniWorldEnv
from my.mesh_ent import MeshEnt
from tap import Tap


class Args(Tap):
    data_path: str = "/root/.cache/data/ycb"
    obj_pattern: str = "*/google_16k/textured.obj"
    png_pattern: str = "*/google_16k/texture_map.png"
    room_size: int = 4


class Mesh(NamedTuple):
    obj: Union[Path, str]
    png: Optional[Path]
    name: str


T = TypeVar("T")  # Declare type variable


@dataclass
class Obs:
    image: T
    mission: T

    def to_space(self):
        return gym.spaces.Dict(**asdict(self))

    def to_obs(self, observation_space: Space = None, check: bool = True):
        obs = asdict(self)
        if observation_space and check:
            assert observation_space.contains(obs), (observation_space, obs)
        return obs


class String(gym.Space):
    def sample(self):
        return "".join(self.np_random.choice(string.ascii_letters) for _ in range(10))

    def contains(self, x):
        return isinstance(x, str)


@dataclass
class Timestep:
    s: Obs = None
    a: MiniWorldEnv.Actions = None
    r: float = None
    t: bool = None
    i: dict = None


OneRoom


class Env(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(
        self,
        meshes: List[Mesh],
        seed: int,
        size: int,
        max_episode_steps: int = 180,
        **kwargs,
    ):
        self.meshes = meshes
        assert size >= 2
        self.size = size
        self.rng = np.random.default_rng(seed=seed)
        self._timestep = Timestep()

        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        self.observation_space = Obs(
            image=self.observation_space, mission=String()
        ).to_space()

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)
        meshes = self.rng.choice(len(self.meshes), size=2, replace=False)
        goal_mesh, _ = meshes = [self.meshes[i] for i in meshes]
        self.mission = goal_mesh.name
        meshes = [
            MeshEnt(
                str(mesh.obj), height=1, tex_name=str(mesh.png) if mesh.png else None
            )
            for mesh in meshes
        ]
        self.goal, self.dist = [self.place_entity(mesh) for mesh in meshes]

        self.place_agent()

    @staticmethod
    def ascii_of_image(image: np.ndarray):
        def rows():
            for row in image:
                yield "".join([(color("██", tuple(rgb.astype(int)))) for rgb in row])

        return "\n".join(rows())

    def make_obs(self, image: np.ndarray) -> Obs:
        return Obs(image=image, mission=self.mission)

    def render(self, mode="human", pause=True, **kwargs):
        if mode == "ascii":
            print(self.ascii_of_image(self.render_obs()))
            print()
            subtitle = self.mission
            if self._timestep.a is not None:
                action = self._timestep.a.name.replace("_", " ")
                subtitle = f"{subtitle}, {action}, r={round(self._timestep.r, 2)}"
                if self._timestep.t:
                    subtitle += ", done"
            print(text2art(subtitle.swapcase(), font="com_sen"))
            if pause:
                input("Press enter to continue.")
        else:
            return super().render(mode=mode, **kwargs)

    def reset(self) -> dict:
        obs = self.make_obs(super().reset())
        self.update_timestep(s=obs)
        return self.to_obs(obs)

    def step(self, action: np.ndarray):
        self.update_timestep(a=[*MiniWorldEnv.Actions][action.item()])
        image, reward, done, info = super().step(action)

        if self.near(self.goal):
            reward += self._reward()
            done = True
        elif self.near(self.dist):
            done = True

        obs = self.make_obs(image)
        self.update_timestep(s=obs, r=reward, t=done, i=info)
        return self.to_obs(obs), reward, done, info

    def to_obs(self, obs: Obs):
        check = isinstance(self.observation_space, gym.spaces.Dict)
        return obs.to_obs(self.observation_space, check=check)

    def update_timestep(
        self,
        s: Obs = None,
        a: MiniWorldEnv.Actions = None,
        r: float = None,
        t: bool = None,
        i: dict = None,
    ):
        mapping = dict(s=s, a=a, r=r, t=t, i=i)
        kwargs = {k: v for k, v in mapping.items() if v is not None}
        self._timestep = replace(self._timestep, **kwargs)
