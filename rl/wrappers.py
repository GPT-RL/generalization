import abc
import re
import typing
from abc import ABC
from dataclasses import astuple, dataclass, replace
from functools import lru_cache
from typing import Generator, Optional, TypeVar

import gym
import gym_minigrid
import numpy as np
from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import ObjDesc, PickupInstr
from colors import color as ansi_color
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import COLORS, OBJECT_TO_IDX, MiniGridEnv, WorldObj
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

from pybullet_env import Observation

T = TypeVar("T")  # Declare type variable


@dataclass
class TrainTest:
    train: list
    test: list


class Agent(WorldObj):
    def render(self, r):
        pass


class ReproducibleEnv(RoomGridLevel, ABC):
    def _rand_elem(self, iterable):
        return super()._rand_elem(sorted(iterable))


class RenderEnv(RoomGridLevel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__reward = None
        self.__done = None
        self.__action = None

    def row_objs(self, y: int) -> Generator[Optional[WorldObj], None, None]:
        for x in range(self.width):
            if np.all(self.agent_pos == (x, y)):
                yield Agent(color="grey", type="agent")
            else:
                yield self.grid.get(x, y)

    def row_strings(self, i: int) -> Generator[str, None, None]:
        for obj in self.row_objs(i):
            if obj is None:
                string = ""
            elif isinstance(obj, Agent):
                if self.agent_dir == 0:
                    string = ">"
                elif self.agent_dir == 1:
                    string = "v"
                elif self.agent_dir == 2:
                    string = "<"
                elif self.agent_dir == 3:
                    string = "^"
                else:
                    breakpoint()
                    raise RuntimeError
            else:
                string = obj.type

            string = f"{string:<{self.max_string_length}}"
            if obj is not None:
                color = obj.color
                string = self.color_obj(color, string)
            yield string + "\033[0m"

    @staticmethod
    def color_obj(color, string):
        return ansi_color(string, tuple(COLORS[color]))

    @property
    def max_string_length(self):
        return max(map(len, OBJECT_TO_IDX)) + 1

    def row_string(self, i: int):
        return "|".join(self.row_strings(i))

    def horizontal_separator_string(self):
        return "-" * ((self.max_string_length + 1) * self.grid.width - 1)

    def render_string(self):
        yield self.row_string(0)
        for i in range(1, self.grid.height):
            yield self.horizontal_separator_string()
            yield self.row_string(i)

    def render(self, mode="human", pause=True, **kwargs):
        if mode == "human":
            for string in self.render_string():
                print(string)
            print(self.mission)
            print("Reward:", self.__reward)
            print("Done:", self.__done)
            print(
                "Action:",
                None
                if self.__action is None
                else MiniGridEnv.Actions(self.__action).name,
            )
            self.pause(pause)
        else:
            return super().render(mode=mode, **kwargs)

    @staticmethod
    def pause(pause):
        if pause:
            input("Press enter to continue.")

    def step(self, action):
        self.__action = action
        s, self.__reward, self.__done, i = super().step(action)
        return s, self.__reward, self.__done, i


class RenderColorEnv(RenderEnv, ABC):
    @staticmethod
    def color_obj(color: str, string: str):
        if re.match("([v^><]|wall) *", string):
            return string
        return color.ljust(len(string))


class PickupEnv(RenderEnv, ReproducibleEnv):
    def __init__(
        self,
        objects: typing.Iterable[typing.Tuple[str, str]],
        room_size: int,
        seed: int,
        strict: bool,
        num_dists: int = 1,
        prohibited=None,
    ):
        self.prohibited = prohibited
        self.objects = sorted(objects)
        self.strict = strict
        self.num_dists = num_dists
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed,
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()

        goal_object = self._rand_elem(self.objects)
        self.add_object(0, 0, *goal_object)
        objects = {*self.objects} - {goal_object}
        for _ in range(self.num_dists):
            obj = self._rand_elem(objects)
            if {goal_object, obj} != self.prohibited:
                self.add_object(0, 0, *obj)

        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(*goal_object), strict=self.strict)


class RenderColorPickupEnv(RenderColorEnv, PickupEnv):
    pass


class MissionWrapper(gym.Wrapper, abc.ABC):
    def __init__(self, env):
        self._mission = None
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._mission = self.change_mission(observation["mission"])
        observation["mission"] = self._mission
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation["mission"] = self._mission
        return observation, reward, done, info

    def render(self, mode="human", pause=True, **kwargs):
        self.env.render(pause=False)
        print(self._mission)
        self.env.pause(pause)

    def change_mission(self, mission: str) -> str:
        raise NotImplementedError


mapping = {
    "green box": "green animal",
    "yellow box": "orange animal",
    "green ball": "green food",
    "yellow ball": "orange food",
    "grey box": "white animal",
    "grey ball": "white food",
    "purple box": "purple animal",
    "purple ball": "purple food",
    "blue box": "black animal",
    "blue ball": "black food",
    "red box": "red animal",
    "red ball": "red food",
}


# noinspection PyShadowingBuiltins
def alt_type(type: str):
    if type == "ball":
        return "box"
    elif type == "box":
        return "ball"
    else:
        raise RuntimeError()


class PlantAnimalWrapper(MissionWrapper):
    green_animal = "green box"
    orange_animal = "yellow box"
    green_plant = "green ball"
    orange_plant = "yellow ball"
    white_animal = "grey box"
    white_plant = "grey ball"
    purple_animal = "purple box"
    purple_plant = "purple ball"
    # pink_animal = "pink box"
    # pink_plant = "pink ball"
    black_animal = "blue box"
    black_plant = "blue ball"
    red_animal = "red box"
    red_plant = "red ball"
    replacements = {
        red_animal: [
            "rooster",
            "lobster",
            "crab",
            "ladybug",
            "cardinal",
        ],
        red_plant: [
            "cherry",
            "tomato",
            "chili",
            "apple",
            "raspberry",
            "cranberry",
            "strawberry",
            "pomegranate",
            "radish",
            "beet",
            "rose",
        ],
        black_animal: [
            "gorilla",
            "crow",
            "panther",
            "raven",
            "bat",
        ],
        black_plant: ["black plant"],
        # pink_animal: ["flamingo", "pig"],
        # pink_plant: ["lychee", "dragonfruit"],
        purple_animal: ["purple animal"],
        purple_plant: [
            "grape",
            "eggplant",
            "plum",
            "shallot",
            "lilac",
        ],
        white_animal: [
            "polar bear",
            "swan",
            "ermine",
            "sheep",
            "seagull",
        ],
        white_plant: [
            "coconut",
            "cauliflower",
            "onion",
            "garlic",
        ],
        green_animal: [
            "iguana",
            "frog",
            "grasshopper",
            "turtle",
            "mantis",
            "lizard",
            "caterpillar",
        ],
        green_plant: [
            "lime",
            "kiwi",
            "broccoli",
            "lettuce",
            "kale",
            "spinach",
            "avocado",
            "cucumber",
            "basil",
            "pea",
            "arugula",
            "celery",
        ],
        orange_animal: [
            "tiger",
            "lion",
            "orangutan",
            "goldfish",
            "clownfish",
            "fox",
        ],
        orange_plant: [
            "peach",
            "yam",
            "tangerine",
            "carrot",
            "papaya",
            "clementine",
            "kumquat",
            "pumpkin",
            "marigold",
        ],
    }

    def __init__(self, env, prefixes: dict):
        self.prefixes = prefixes
        self.missions = [self.new_mission(m) for m in self.replacements]
        super().__init__(env)

    def change_mission(self, mission: str) -> str:
        mission = mission.replace("pick up the ", "")
        mission = self.new_mission(mission)
        return mission

    def new_mission(self, mission):
        color, type = mission.split()
        prefix = self.prefixes[type]
        if prefix != "":
            mission = f"{prefix}. {color} {alt_type(type)}: {color}"
        return mission


class ActionInObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.spaces = dict(
            **self.observation_space.spaces, action=Discrete(self.action_space.n + 1)
        )
        self.observation_space = Dict(spaces=self.spaces)

    def reset(self, **kwargs):
        s = super().reset(**kwargs)
        s["action"] = self.spaces["action"].n - 1
        return s

    def step(self, action):
        s, r, t, i = super().step(action)
        s["action"] = action
        return s, r, t, i


class SuccessWrapper(gym.Wrapper):
    def step(self, action):
        s, r, t, i = super().step(action)
        if t:
            i.update(success=r > 0)
        return s, r, t, i


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = Observation(*self.observation_space.spaces)
        self.original_observation_space = Tuple(
            astuple(Observation(*self.observation_space.spaces))
        )

        def sizes():
            for space in astuple(spaces):
                if isinstance(space, Box):
                    yield np.prod(space.shape)
                elif isinstance(space, MultiDiscrete):
                    yield space.nvec.size
                elif isinstance(space, Discrete):
                    yield 1

        self.observation_space = Box(
            shape=[sum(sizes())],
            low=-np.inf,
            high=np.inf,
        )

    def observation(self, observation):
        obs = np.concatenate(astuple(Observation(*[x.flatten() for x in observation])))
        # assert self.observation_space.contains(obs)
        return obs


class TokenizerWrapper(gym.ObservationWrapper):
    def __init__(self, env, tokenizer: GPT2Tokenizer):
        self.tokenizer: GPT2Tokenizer = tokenizer
        super().__init__(env)
        tokens = [self.encode(m, tokenizer) for m in env.missions]
        ns, ds = zip(*(t.shape for t in tokens))
        maxes = [int(t.max()) for t in tokens]
        nvec = max(maxes) * np.ones((max(ns), max(ds)))
        spaces = Observation(*self.observation_space.spaces)
        self.observation_space = Tuple(
            astuple(
                replace(
                    spaces,
                    mission=MultiDiscrete(nvec),
                )
            )
        )

    @staticmethod
    @lru_cache()
    def encode(mission, tokenizer):
        words = mission.split()
        tokens = [tokenizer.encode(w, return_tensors="pt").T for w in words]
        padded = (
            pad_sequence(tokens, padding_value=tokenizer.eos_token_id).squeeze(-1).T
        )
        return padded

    def observation(self, observation):
        mission = self.new_mission(
            self.tokenizer,
            observation["mission"],
            tuple(self.observation_space.spaces["mission"].nvec.shape),
        )
        observation.update(mission=mission)
        return observation

    @classmethod
    @lru_cache()
    def new_mission(
        cls,
        tokenizer: GPT2Tokenizer,
        mission: str,
        mission_shape: typing.Tuple[int, int],
    ):
        encoded = cls.encode(mission, tokenizer).numpy()
        n1, d1 = encoded.shape
        n2, d2 = mission_shape
        assert n2 >= n1 and d2 >= d1
        padded = np.pad(
            encoded,
            [(0, n2 - n1), (0, d2 - d1)],
            constant_values=tokenizer.eos_token_id,
        )
        # assert mission_space.contains(padded)
        return padded


class MissionEnumeratorWrapper(gym.ObservationWrapper):
    def __init__(self, env, missions: typing.Iterable[str]):
        self.missions = list(missions)
        self.cache = {k: i for i, k in enumerate(self.missions)}
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.observation_space = Dict(
            spaces=dict(**spaces, mission=Discrete(len(self.cache)))
        )

    def observation(self, observation):
        observation.update(mission=self.cache[observation["mission"]])
        return observation


class DirectionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Dict(
            spaces=dict(
                **self.observation_space.spaces,
                direction=Discrete(4),
            )
        )


class FullyObsWrapper(gym_minigrid.wrappers.FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Dict(
            spaces=dict(
                **self.observation_space.spaces,
                direction=Discrete(4),
            )
        )

    def observation(self, obs):
        direction = obs["direction"]
        obs = super().observation(obs)
        obs["direction"] = direction
        return obs


def main(args: "Args"):
    def redraw(img):
        if not args.agent_view:
            img = env.render("rgb_array", tile_size=args.tile_size)
        window.show_img(img)

    def reset():
        obs = env.reset()

        if hasattr(env, "mission"):
            print("Mission: %s" % env.mission)
            window.set_caption(env.mission)

        redraw(obs)

    def step(action):
        obs, reward, done, info = env.step(action)
        print(f"step={env.step_count}, reward={reward}, success={info.get('success')}")

        if done:
            print("done!")
            reset()
        else:
            redraw(obs)

    def key_handler(event):
        print("pressed", event.key)

        if event.key == "escape":
            window.close()
            return

        if event.key == "backspace":
            reset()
            return

        if event.key == "left":
            step(env.actions.left)
            return
        if event.key == "right":
            step(env.actions.right)
            return
        if event.key == "up":
            step(env.actions.forward)
            return

        # Spacebar
        if event.key == " ":
            step(env.actions.toggle)
            return
        if event.key == "pageup":
            step(env.actions.pickup)
            return
        if event.key == "pagedown":
            step(env.actions.drop)
            return

        if event.key == "enter":
            step(env.actions.done)
            return

    room_objects = [("ball", col) for col in ("black", "white")]
    env = PickupEnv(
        room_size=args.room_size,
        seed=args.seed,
        objects=room_objects,
        strict=True,
    )
    if args.agent_view:
        env = ImgObsWrapper(env)
    window = Window("gym_minigrid")
    window.reg_key_handler(key_handler)
    reset()
    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    import pybullet_main

    class Args(pybullet_main.Args):
        tile_size: int = 32
        agent_view: bool = False
        test: bool = False
        not_strict: bool = False
        num_dists: int = 1

    main(Args().parse_args())