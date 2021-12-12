import typing
from dataclasses import dataclass
from typing import Generator, List, Literal, TypeVar

import babyai.utils as utils
import gym
import gym_minigrid
import numpy as np
import torch
from gym.spaces import Dict, Discrete
from tap import Tap
from transformers import pipeline


class Args(Tap):
    buffer_size: int = 10_000_000
    demos: str = None
    env: str = "BabyAI-GoToRedBall-v0"
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
    num_episodes: int = 30
    oracle_model: str = "BOT"


T = TypeVar("T")  # Declare type variable


@dataclass
class Obs:
    image: T
    direction: T
    mission: T


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


def main(args: Args):
    env = gym.make(args.env)
    rng = np.random.Generator()
    agent = utils.load_agent(
        env=env,
        model_name=args.oracle_model,
        demos_origin=args.demos,
        env_name=args.env,
    )
    generator = pipeline("text-generation", model=args.foundation_model)

    def get_episode() -> Generator[typing.Tuple[Obs, int], None, None]:
        done = False
        obs = env.reset()
        agent.on_reset()
        while not done:
            action = agent.act(obs)["action"]
            if isinstance(action, torch.Tensor):
                action = action.item()
            yield Obs(**obs), action

            obs, _, done, _ = env.step(action)

    def step_to_string(obs: Obs, action: int) -> str:
        breakpoint()
        return ""

    def get_strings() -> Generator[str, None, None]:
        for _ in range(args.num_episodes):
            for obs, action in get_episode():
                yield step_to_string(obs, action)

    generator("EleutherAI has", do_sample=True, min_length=50)

    while True:
        strings: List[str] = list(get_strings())
        rng.shuffle(strings)
        ";".join(strings)


if __name__ == "__main__":
    main(Args().parse_args())
