import logging
from typing import cast

import babyai_main
import gym
import torch
from gpt_agent import Agent


class Args(babyai_main.Args):
    randomize_parameters: bool = False
    train_ln: bool = False
    train_wpe: bool = False
    gpt: bool = False


class ArgsType(Args, babyai_main.ArgsType):
    pass


class Trainer(babyai_main.Trainer):
    @classmethod
    def _make_agent(
        cls,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        args: ArgsType,
    ):
        return Agent(
            action_space=action_space,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
            pretrained_model=args.pretrained_model,
            randomize_parameters=args.randomize_parameters,
            train_wpe=args.train_wpe,
            train_ln=args.train_ln,
            use_gru=args.use_gru,
        )

    @staticmethod
    def save(
        agent: Agent,
        save_path,
        args,
    ):
        trainable = {k for k, v in agent.named_parameters() if v.requires_grad}
        trainable_params = {
            k: v for k, v in agent.state_dict().items() if k in trainable
        }
        torch.save(trainable_params, save_path)
        logging.info(f"Saved the following modules to {save_path}:")
        for p in trainable_params:
            logging.info(p)

    @staticmethod
    def load(agent, load_path):
        loaded = torch.load(load_path)
        for name, module in agent.named_modules():
            name_ = f"{name}."
            parameters = {}
            for k, v in loaded.items():
                if k.startswith(name_):
                    parameters[k[len(name_) :]] = v
            if parameters:
                try:
                    module.load_state_dict(parameters)
                    logging.info(f"Loaded parameters into {name}.")
                except RuntimeError:
                    pass

    @classmethod
    def train(cls, args: Args, **kwargs):
        return (
            super().train(args, **kwargs)
            if args.gpt
            else babyai_main.Trainer().train(args, **kwargs)
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
