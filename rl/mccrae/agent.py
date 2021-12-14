from typing import Literal

import agent
import gym
import numpy as np
import torch
from agent import NNBase
from mccrae.env import Obs, get_size
from torch import nn
from transformers import (
    BertConfig,
    BertTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    GPTNeoConfig,
)
from utils import init

BINARY_PRETRAINED = "binary-pretrained"
BINARY_UNTRAINED = "binary-untrained"
PRETRAINED = "pretrained"
UNTRAINED = "untrained"
BASELINE = "baseline"
# noinspection PyTypeHints
Architecture = Literal[
    BINARY_PRETRAINED, BINARY_UNTRAINED, PRETRAINED, UNTRAINED, BASELINE
]


def make_tokenizer(pretrained_model):
    if "gpt" in pretrained_model:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    elif "bert" in pretrained_model:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    else:
        raise RuntimeError(f"Invalid model name: {pretrained_model}")
    return tokenizer


def build_gpt(gpt_size: str, randomize_parameters):
    return (
        GPT2Model(
            GPT2Config.from_pretrained(
                gpt_size,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        )
        if randomize_parameters
        else GPT2Model.from_pretrained(
            gpt_size,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    )


ModelName = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def get_embedding_size(model_name: ModelName):
    if "neo" in model_name:
        config = GPTNeoConfig.from_pretrained(
            model_name,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        return config.hidden_size
    elif "gpt2" in model_name:
        config = GPT2Config.from_pretrained(
            model_name,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        return config.n_embd
    elif "bert" in model_name:
        config = BertConfig.from_pretrained(
            model_name,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        return config.hidden_size

    else:
        raise RuntimeError(f"Invalid model name: {model_name}")


class GPTEmbed(nn.Module):
    def __init__(
        self,
        model_name: str,
        randomize_parameters: bool,
        train_wpe: bool,
        train_ln: bool,
    ):
        super().__init__()
        tokenizer = make_tokenizer(model_name)
        self.pad_token_id = tokenizer.eos_token_id
        self.gpt = build_gpt(model_name, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

    def forward(self, x, **_):
        return self.gpt.forward(
            x, attention_mask=x != self.pad_token_id
        ).last_hidden_state[:, -1]


class Base(NNBase):
    def __init__(
        self,
        architecture: Architecture,
        model_name: ModelName,
        hidden_size: int,
        observation_space: gym.spaces.Tuple,
        recurrent: bool,
        **kwargs,
    ):
        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.observation_spaces = Obs(*observation_space.spaces)
        embedding_size = get_embedding_size(model_name)
        self.encode_mission = (
            nn.EmbeddingBag(
                int(self.observation_spaces.mission.nvec.max()) + 1, embedding_size
            )
            if architecture == BASELINE
            else GPTEmbed(
                model_name=model_name,
                randomize_parameters=architecture == UNTRAINED,
                **kwargs,
            )
        )

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        image_shape = self.observation_spaces.image.shape
        if len(image_shape) == 3:
            h, w, d = image_shape
            dummy_input = torch.zeros(1, d, h, w)

            self.image_net = nn.Sequential(
                init_(nn.Conv2d(d, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            try:
                output = self.image_net(dummy_input)
            except RuntimeError:
                self.image_net = nn.Sequential(
                    init_(nn.Conv2d(d, 32, 3, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                output = self.image_net(dummy_input)
        else:
            dummy_input = torch.zeros(image_shape)
            self.image_net = nn.Sequential(
                nn.Linear(int(np.prod(image_shape)), hidden_size), nn.ReLU()
            )
            output = self.image_net(dummy_input)

        self.merge = nn.Sequential(
            init_(
                nn.Linear(
                    output.size(-1) + embedding_size,
                    hidden_size,
                )
            ),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def forward(self, inputs: torch.Tensor, rnn_hxs: torch.Tensor, masks: torch.Tensor):
        inputs = Obs(
            *torch.split(
                inputs,
                [get_size(space) for space in self.observation_spaces],
                dim=-1,
            )
        )

        image = inputs.image.reshape(-1, *self.observation_spaces.image.shape)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        image = self.image_net(image)

        mission = self.encode_mission(inputs.mission.long())
        x = torch.cat([image, mission], dim=-1)
        x = self.merge(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs


class Agent(agent.Agent):
    def __init__(self, observation_space, **kwargs):
        super().__init__(
            obs_shape=observation_space.spaces.image.shape,
            observation_space=observation_space,
            **kwargs,
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)
