import logging
from typing import cast

import babyai_main
import torch
from babyai_env import Spaces, TokenizerWrapper
from envs import VecPyTorch
from gpt_agent import Agent, Base
from utils import build_gpt


class Args(babyai_main.Args):
    multihead_attention: bool = False
    freeze_keys: bool = False
    randomize_parameters: bool = False
    attn_temp: float = 5
    train_ln: bool = False
    train_wpe: bool = False
    gpt: bool = False


class ArgsType(Args, babyai_main.ArgsType):
    pass


class Trainer(babyai_main.Trainer):
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
        action_space = envs.action_space
        observation_space, *_ = envs.get_attr("original_observation_space")
        gpt = build_gpt(args.pretrained_model, args.randomize_parameters)
        outputs = None
        if args.multihead_attention:
            tokenizer = cls.tokenizer(args.pretrained_model)
            missions, *_ = envs.get_attr("missions")
            mission_space = Spaces(*observation_space.spaces).mission
            tokens = [
                TokenizerWrapper.new_mission(tokenizer, mission, mission_space)
                for mission in missions
            ]
            tokens = torch.Tensor(tokens).long()
            outputs = Base.pass_through_gpt(gpt, tokens, tokenizer.eos_token_id)
            outputs = outputs.reshape(-1, outputs.size(-1))

        return Agent(
            action_space=action_space,
            attn_temp=args.attn_temp,
            freeze_keys=args.freeze_keys,
            gpt=gpt,
            hidden_size=args.hidden_size,
            multihead_attention=args.multihead_attention,
            observation_space=observation_space,
            outputs=outputs,
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
            else babyai_main.Trainer().train(args, **kwargs)
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
