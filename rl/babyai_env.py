import abc
import re
import typing
from abc import ABC
from dataclasses import astuple, dataclass
from itertools import chain, cycle, islice
from typing import Generator, Optional, TypeVar

import gym
import gym_minigrid
import numpy as np
from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import (
    ObjDesc,
    PickupInstr,
)
from colors import color as ansi_color
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import COLORS, MiniGridEnv, OBJECT_TO_IDX, WorldObj
from gym_minigrid.window import Window
from gym_minigrid.wrappers import (
    ImgObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
)
from transformers import GPT2Tokenizer

T = TypeVar("T")  # Declare type variable


@dataclass
class Spaces:
    image: T
    direction: T
    mission: T
    action: T


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


class PickupEnv(ReproducibleEnv, RenderEnv):
    def __init__(
        self,
        room_objects: typing.Iterable[typing.Tuple[str, str]],
        room_size: int,
        seed: int,
        strict: bool,
        num_dists: int = 1,
    ):
        self.strict = strict
        self.room_objects = list(room_objects)
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
        goal_object = self._rand_elem(self.room_objects)
        self.add_object(0, 0, *goal_object)
        objects = {*self.room_objects} - {goal_object}
        for _ in range(self.num_dists):
            obj = self._rand_elem(objects)
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

    def change_mission(self, mission: str) -> str:
        for k, v in self.replacements.items():
            if k in mission:
                replacement = self.np_random.choice(v)
                mission = mission.replace(k, replacement)

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


class ZeroOneRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return int(bool(reward > 0))


class RolloutsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.original_observation_space = Tuple(
            astuple(Spaces(**self.observation_space.spaces))
        )
        image_space = spaces["image"]
        mission_space = spaces["mission"]
        self.observation_space = Box(
            shape=[np.prod(image_space.shape) + 2 + np.prod(mission_space.shape)],
            low=-np.inf,
            # direction_space = spaces["direction"]
            high=np.inf,
        )

    def observation(self, observation):
        return np.concatenate(
            astuple(
                Spaces(
                    image=observation["image"].flatten(),
                    direction=np.array([observation["direction"]]),
                    action=np.array([int(observation["action"])]),
                    mission=observation["mission"],
                )
            )
        )


class TokenizerWrapper(gym.ObservationWrapper):
    def __init__(self, env, tokenizer: GPT2Tokenizer, longest_mission: str):
        self.tokenizer: GPT2Tokenizer = tokenizer
        encoded = tokenizer.encode(longest_mission)
        super().__init__(env)
        spaces = {**self.observation_space.spaces}
        self.observation_space = Dict(
            spaces=dict(**spaces, mission=MultiDiscrete([50257 for _ in encoded]))
        )

    def observation(self, observation):
        mission = self.tokenizer.encode(observation["mission"])
        length = len(self.observation_space.spaces["mission"].nvec)
        eos = self.tokenizer.eos_token_id
        mission = [*islice(chain(mission, cycle([eos])), length)]
        observation.update(mission=mission)
        return observation


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


class RGBImgObsWithDirectionWrapper(RGBImgObsWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

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
        room_objects=room_objects,
        strict=True,
    )
    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
    window = Window("gym_minigrid")
    window.reg_key_handler(key_handler)
    reset()
    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    import babyai_main

    class Args(babyai_main.Args):
        tile_size: int = 32
        agent_view: bool = False
        test: bool = False
        not_strict: bool = False
        num_dists: int = 1

    main(Args().parse_args())
