import itertools
import re
import string as python_string
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generator, Literal

import babyai.utils as utils
import gym
import torch
from gym_minigrid.minigrid import COLORS
from tap import Tap
from transformers import pipeline


class Args(Tap):
    demos: str = None
    env: str = "BabyAI-GoToObj-v0"
    foundation_model: Literal[
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "bert-base-uncased",
        "bert-large-uncased",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
    ] = "EleutherAI/gpt-neo-2.7B"
    oracle_model: str = "BOT"
    seed: int = 0


OBJECT_TO_STR = OrderedDict(
    {
        "wall": "W",
        "floor": "F",
        "door": "D",
        "key": "K",
        "ball": "A",
        "box": "B",
        "goal": "G",
        "lava": "V",
    }
)

AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}


class StringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        def pairs():
            for o in OBJECT_TO_STR.values():
                for (c, *_) in COLORS:
                    yield o, c.upper()

        def directions():
            for direction in AGENT_DIR_TO_STR.values():
                yield (direction, direction), direction

        self.pair_to_idx = {
            **dict(zip(pairs(), python_string.printable)),
            **dict(directions()),
            (" ", " "): " ",
        }

    def __str__(self):
        original = super().__str__()
        matches = re.match(
            "<" + self.class_name() + r"(.*)>", original, flags=re.DOTALL
        )
        lines = matches.group(1).split("\n")

        def replace_line(line: str):
            if not line:
                return ""
            c1, c2, *line = line
            idx = self.pair_to_idx[c1, c2]
            replaced = replace_line(line)
            return f"{idx}{replaced}"

        return "\n".join(map(replace_line, lines))


@dataclass
class Step:
    action: int
    image: str
    mission: str


def main(args: Args):
    env = gym.make(args.env)
    env.seed(args.seed)
    env = StringWrapper(env)
    # rng = np.random.default_rng(args.seed)
    print("Loading agent...")
    agent = utils.load_agent(
        env=env,
        model_name=args.oracle_model,
        demos_origin=args.demos,
        env_name=args.env,
    )
    print("Building pipeline...")
    generator = pipeline("text-generation", model=args.foundation_model)

    def get_steps() -> Generator[Step, None, None]:
        done = False
        obs = env.reset()
        agent.on_reset()
        while not done:
            action = agent.act(obs)["action"]
            if isinstance(action, torch.Tensor):
                action = action.item()

            yield Step(mission=obs["mission"], image=str(env), action=action)

            obs, _, done, _ = env.step(action)

    AGENT_ACTION_TO_STR = {0: "l", 1: "r", 2: "f", 3: "p", 4: "d", 5: "t", 6: "d"}

    def partial_step_to_string(step: Step) -> str:
        return f"{step.image}"
        # return f"{step.image}{step.mission}"

    def step_to_string(step: Step) -> str:
        return f"{partial_step_to_string(step)}\n{AGENT_ACTION_TO_STR[step.action]}"

    total = 2048
    for seed in itertools.count():
        print("Seed:", seed)
        env.seed(seed + args.seed)

        def get_prefix_parts():
            length = 0
            while True:
                for step in get_steps():
                    full_string = step_to_string(step)
                    partial_string = partial_step_to_string(step)
                    length += len(full_string)
                    if length > total:
                        return
                    yield full_string, partial_string, step

        (*strings, _, _), (*partials, partial, _), (*_, last_step, _) = zip(
            *get_prefix_parts()
        )
        *_, second_to_last_string = strings
        print(second_to_last_string)
        print(partial)
        print("Correct action is:", AGENT_ACTION_TO_STR[last_step.action])
        assert len(strings) == len(partials)
        prefix = "\n".join([*strings, partial])
        print("Generating...")
        generated = generator(
            prefix,
            do_sample=False,
            min_length=len(prefix) + 1,
            max_length=len(prefix) + 1,
        )
        output = generated[0]["generated_text"][len(prefix) + 1]
        print("Agent output:", output)
        print("Correct inference:", AGENT_ACTION_TO_STR[last_step.action] == output)


if __name__ == "__main__":
    main(Args().parse_args())
