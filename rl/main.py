import logging
from typing import cast

import gym
import my.main
import torch
from gpt_agent import Agent
from gym.spaces import Discrete


class Args(my.main.Args):
    gpt: bool = False
    randomize_parameters: bool = False
    train_ln: bool = False
    train_wpe: bool = False


class ArgsType(Args, my.main.ArgsType):
    pass


class Trainer(my.main.Trainer):
    @classmethod
    def _make_agent(
        cls,
        action_space: Discrete,
        args: Args,
        device: torch.device,
        missions: list,
        observation_space: gym.spaces.Dict,
    ):
        return Agent(
            action_space=action_space,
            attn_temp=args.attn_temp,
            device=device,
            freeze_keys=args.freeze_keys,
            hidden_size=args.hidden_size,
            multihead_attention=args.multihead_attention,
            observation_space=observation_space,
            missions=missions,
            pretrained_model=args.pretrained_model,
            randomize_parameters=args.randomize_parameters,
            recurrent=cls.recurrent(args),
            train_wpe=args.train_wpe,
            train_ln=args.train_ln,
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
            else my.main.Trainer().train(args, **kwargs)
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
