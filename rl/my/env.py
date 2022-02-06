import string
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, NamedTuple, Optional, Set, TypeVar, Union

import gym
import gym_miniworld
import numpy as np
import pandas as pd
from art import text2art
from colors import color
from gym import Space, spaces
from gym_miniworld.envs import OneRoom
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.params import DEFAULT_PARAMS
from my.mesh_ent import MeshEnt
from tap import Tap


class Args(Tap):
    data_path: str = Path(Path.home(), ".cache/data/ycb")
    names: Optional[str] = None
    room_size: int = 8


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
        size: int,
        max_episode_steps: int = 180,
        pitch: float = 0,
        rank: int = 0,
        **kwargs,
    ):
        self.rank = rank
        assert size >= 2
        self.size = size
        self._timestep = Timestep()

        self.meshes = meshes

        params = deepcopy(DEFAULT_PARAMS)
        params.set("cam_pitch", pitch, pitch, pitch)
        super().__init__(max_episode_steps=max_episode_steps, params=params, **kwargs)
        # Allow only movement actions (left/right/forward) and pickup
        self.action_space = spaces.Discrete(self.actions.pickup + 1)
        self.observation_space = Obs(
            image=self.observation_space, mission=String()
        ).to_space()

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)
        meshes = self.rand.subset(self.meshes, num_elems=2)
        self.mission, _ = [m.name for m in meshes]
        self.goal, self.dist = [
            self.place_entity(
                MeshEnt(
                    str(mesh.obj),
                    height=0.8,
                    static=False,
                    tex_name=str(mesh.png) if mesh.png else None,
                )
            )
            for mesh in meshes
        ]

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

    def step(self, action: Union[np.ndarray, MiniWorldEnv.Actions]):
        a = (
            action
            if isinstance(action, MiniWorldEnv.Actions)
            else [*MiniWorldEnv.Actions][action.item()]
        )
        self.update_timestep(a=a)
        image, reward, done, info = super().step(action)

        if self.agent.carrying == self.goal:
            reward += self._reward()
            done = True
        elif self.agent.carrying == self.dist:
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


def get_meshes(
    data_path: Optional[str],
    names: Optional[str],
):
    if names:
        names: Set[str] = set(names.split(","))
    default_meshes_dir = Path(Path(gym_miniworld.__file__).parent, "meshes")
    if data_path is not None:
        data_path = Path(data_path).expanduser()
        if not data_path.exists():
            raise RuntimeError(
                f"""\
    {data_path} does not exist.
    Download dataset using https://github.com/sea-bass/ycb-tools
    """
            )

    def _get_meshes():
        for _, row in pd.read_csv("ycb.csv").iterrows():
            path = Path(row.path)
            assert path.name == "textured.obj"
            parent = Path(data_path, path.parent)
            obj = Path(parent, path.stem)
            png = Path(parent, "texture_map.png")
            height = row.get("height", 1)
            yield Mesh(obj=obj, png=png, name=row["name"], height=height)

    data_path_meshes = list(_get_meshes())

    default_meshes = [
        Mesh(height=1, name=name.replace("_", " "), obj=name, png=None)
        for name in {m.stem for m in default_meshes_dir.iterdir()}
    ]
    if names is None:
        meshes = default_meshes if data_path is None else data_path_meshes
    else:
        data_path_meshes = {m.name: m for m in data_path_meshes}
        default_meshes = {m.name: m for m in default_meshes}

        def _get_meshes():
            for name in names:
                if name in data_path_meshes:
                    yield data_path_meshes[name]
                elif name in default_meshes:
                    yield default_meshes[name]
                else:
                    raise RuntimeError(f"Invalid name: {name}")

        meshes = list(_get_meshes())
    return meshes
