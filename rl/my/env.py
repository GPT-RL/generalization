import math
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, NamedTuple, TypeVar

import gym
import numpy as np
from gym import Space, spaces
from gym_miniworld.miniworld import MiniWorldEnv
from my.mesh_ent import MeshEnt
from tap import Tap


class Mesh(NamedTuple):
    obj: Path
    png: Path
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
            MeshEnt(str(mesh.obj), height=1, tex_name=str(mesh.png)) for mesh in meshes
        ]
        self.goal, _ = [self.place_entity(mesh) for mesh in meshes]

        self.place_agent()

    def make_obs(self, image: np.ndarray) -> dict:
        check = isinstance(self.observation_space, gym.spaces.Dict)
        return Obs(image=image, mission=self.mission).to_obs(
            self.observation_space, check=check
        )

    def reset(self) -> dict:
        return self.make_obs(super().reset())

    def step(self, action):
        image, reward, done, info = super().step(action)

        if self.near(self.goal):
            reward += self._reward()
            done = True

        obs = self.make_obs(image)
        return obs, reward, done, info


class Args(Tap):
    data_path: str = "/root/.cache/data/ycb"
    obj_pattern: str = "*/google_16k/textured.obj"
    png_pattern: str = "*/google_16k/texture_map.png"
    room_size: int = 4


class ManualControlArgs(Args):
    domain_rand: bool = False
    no_time_limit: bool = False
    top_view: bool = False


if __name__ == "__main__":
    import sys
    import time

    import pyglet
    from pyglet.window import key

    args: ManualControlArgs = ManualControlArgs().parse_args()

    env = Env(
        meshes=[
            Mesh(
                obj=Path(f"~/.cache/data/ycb/{name}/google_16k/textured").expanduser(),
                png=Path(
                    f"~/.cache/data/ycb/{name}/google_16k/texture_map"
                ).expanduser(),
                name=re.sub(r"\d+_", "", name),
            )
            for name in [
                "013_apple",
                "014_lemon",
            ]
        ],
        seed=0,
        size=5,
    )
    if args.no_time_limit:
        env.max_episode_steps = math.inf
    if args.domain_rand:
        env.domain_rand = True

    view_mode = "top" if args.top_view else "agent"

    env.reset()

    # Create the display window
    env.render("pyglet", view=view_mode)

    def step(action):
        print(
            "step {}/{}: {}".format(
                env.step_count + 1, env.max_episode_steps, env.actions(action).name
            )
        )

        obs, reward, done, info = env.step(action)

        if reward > 0:
            print("reward={:.2f}".format(reward))

        if done:
            print("done!")
            tick = time.time()
            env.reset()
            print(time.time() - tick)

        env.render("pyglet", view=view_mode)

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print("RESET")
            env.reset()
            env.render("pyglet", view=view_mode)
            return

        if symbol == key.ESCAPE:
            env.close()
            sys.exit(0)

        if symbol == key.UP:
            step(env.actions.move_forward)
        elif symbol == key.DOWN:
            step(env.actions.move_back)

        elif symbol == key.LEFT:
            step(env.actions.turn_left)
        elif symbol == key.RIGHT:
            step(env.actions.turn_right)

        elif symbol == key.PAGEUP or symbol == key.P:
            step(env.actions.pickup)
        elif symbol == key.PAGEDOWN or symbol == key.D:
            step(env.actions.drop)

        elif symbol == key.ENTER:
            step(env.actions.done)

    @env.unwrapped.window.event
    def on_key_release(symbol, modifiers):
        pass

    @env.unwrapped.window.event
    def on_draw():
        env.render("pyglet", view=view_mode)

    @env.unwrapped.window.event
    def on_close():
        pyglet.app.exit()

    # Enter main event loop
    pyglet.app.run()

    env.close()
