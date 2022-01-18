import abc
import re
import sys
from abc import ABC
from dataclasses import astuple, dataclass
from itertools import chain, cycle, islice
from typing import Generator, Optional, Set, TypeVar

import gym
import gym_minigrid
import numpy as np
from babyai.levels.levelgen import LevelGen, RejectSampling
from babyai.levels.verifier import ActionInstr
from colors import color as ansi_color
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from gym_minigrid.minigrid import COLORS, OBJECT_TO_IDX, MiniGridEnv, WorldObj
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper
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


class ReproducibleEnv(LevelGen, ABC):
    def _rand_elem(self, iterable):
        try:
            iterable = sorted(iterable)
        except TypeError:
            pass
        return super()._rand_elem(iterable)


class RenderEnv(LevelGen, ABC):
    def __init__(self, *args, **kwargs):
        self.__reward = None
        self.__done = None
        self.__action = None
        super().__init__(*args, **kwargs)

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


class BabyAIEnv(ReproducibleEnv, RenderEnv, LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.
    Competencies: PickUp, Loc. No unblocking.
    """

    def __init__(
        self,
        action_kinds: str,
        instr_kinds: str,
        locations: bool,
        locked_room_prob: float,
        num_dists: int,
        num_rows: int,
        room_size: int,
        seed: int,
        unblocking: bool,
        train_instructions: Set[ActionInstr],
        test: bool,
        **_,
    ):
        self.train = not test
        self.train_instructions = train_instructions
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            action_kinds=action_kinds.split(","),  # ["pickup"],
            instr_kinds=instr_kinds.split(","),  # ["action"],
            locations=locations,  # True,
            locked_room_prob=locked_room_prob,  # 0,
            num_cols=1,
            num_dists=num_dists,  # 1,
            num_rows=num_rows,
            room_size=room_size,
            seed=seed,
            unblocking=unblocking,  # False,
        )

    def instr_is_valid(self, instr: ActionInstr):
        try:
            self.validate_instrs(instr)
            instr.surface(self)
            return True
        except (RejectSampling, AssertionError):
            return False

    def rand_instr(self, *args, allow_invalid=False, **kwargs):
        if allow_invalid:
            return super().rand_instr(*args, **kwargs)

        while True:
            if self.train:
                valid_train_instructions = list(
                    filter(self.instr_is_valid, self.train_instructions)
                )
                if not valid_train_instructions:
                    print(".", end="")
                    sys.stdout.flush()
                    raise RejectSampling("No valid train instructions.")
                return self._rand_elem(valid_train_instructions)
            else:
                instr = super().rand_instr(*args, **kwargs)
                if instr not in self.train_instructions:
                    return instr


class RenderColorPickupEnv(RenderColorEnv, BabyAIEnv):
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
        spaces = {**self.observation_space.spaces}
        self.original_observation_space = Tuple(
            astuple(Spaces(**self.observation_space.spaces))
        )

        def sizes():
            for space in spaces.values():
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

    env = BabyAIEnv(**args.as_dict())
    if args.agent_view:
        env = ImgObsWrapper(env)
    window = Window("gym_minigrid")
    window.reg_key_handler(key_handler)
    reset()
    # Blocking event loop
    window.show(block=True)


if __name__ == "__main__":
    import babyai_main

    class Args(babyai_main.Args):
        agent_view: bool = False
        tile_size: int = 32

    main(Args().parse_args())
