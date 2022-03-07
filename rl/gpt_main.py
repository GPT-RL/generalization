import logging
from typing import cast

import torch
from gpt_agent import Agent
from my import main


class Args(main.Args):
    randomize_parameters: bool = False
    gpt: bool = False


class ArgsType(Args, main.ArgsType):
    pass


class Trainer(main.Trainer):
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
    def _make_agent(cls, *_args, args: Args, agent_class=Agent, **kwargs):
        return super()._make_agent(
            *_args,
            agent_class=Agent,
            args=args,
            randomize_parameters=args.randomize_parameters,
            **kwargs,
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

    @classmethod
    def train(cls, args: Args, **kwargs):
        return (
            super().train(args, **kwargs)
            if args.gpt
            else main.Trainer().train(args, **kwargs)
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
