from dataclasses import astuple

import agent
import numpy as np
import torch
import torch.nn as nn
from agent import NNBase
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from my.env import Obs
from transformers import GPT2Tokenizer
from utils import init


def get_size(space: Space):
    if isinstance(space, (Box, MultiDiscrete)):
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return 1
    raise TypeError()


class Agent(agent.Agent):
    def __init__(self, observation_space, **kwargs):
        spaces = Obs(**observation_space.spaces)
        super().__init__(
            obs_shape=spaces.image.shape, observation_space=observation_space, **kwargs
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=(3, 3), stride=(1, 1), padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=(3, 3), stride=(1, 1), padding="same"
            ),
        )

    def forward(self, x):
        return x + self.net(x)


class Base(NNBase):
    def __init__(
        self,
        pretrained_model: str,
        hidden_size: int,
        observation_space: Dict,
        recurrent: bool,
        mission_size: int = 64,
    ):

        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=256 + mission_size,
            hidden_size=hidden_size,
        )
        self.observation_spaces = Obs(**observation_space.spaces)

        self.pad_token_id = GPT2Tokenizer.from_pretrained(pretrained_model).eos_token_id

        self.embeddings = self.build_embeddings()

        image_shape = self.observation_spaces.image.shape
        h, w, d = image_shape

        def get_image_net():
            prev = d
            for i, (num_ch, num_blocks) in enumerate(
                [(16, 2), (32, 2), (32, 2)]
            ):  # Downscale.
                yield nn.Conv2d(prev, num_ch, kernel_size=(3, 3), stride=(1, 1))
                yield nn.MaxPool2d(
                    kernel_size=(3, 3),
                    stride=[2, 2],
                )

                # Residual block(s).
                for j in range(num_blocks):
                    yield ResidualBlock(num_ch)
                prev = num_ch

            yield nn.ReLU()
            yield nn.Flatten()

        self.image_net = nn.Sequential(*get_image_net())
        dummy_input = torch.zeros(1, d, h, w)
        output = self.image_net(dummy_input)
        self.image_linear = nn.Sequential(nn.Linear(output.size(-1), 256), nn.ReLU())

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self.initial_hxs = nn.Parameter(self._initial_hxs)

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def build_embeddings(self):
        num_embeddings = 1 + self.pad_token_id
        return GRUEmbed(num_embeddings)

    def embed(self, inputs):
        *shape_, _ = inputs.shape
        inputs = inputs.reshape(-1, inputs.size(-1))
        outputs = self.embeddings.forward(inputs)
        outputs = outputs.reshape(*shape_, -1)
        if outputs.size(1) > 1:
            outputs = outputs.mean(1)
        else:
            outputs = outputs.squeeze(1)
        return outputs

    def forward(self, inputs, rnn_hxs, masks):
        inputs = Obs(
            *torch.split(
                inputs,
                [get_size(space) for space in astuple(self.observation_spaces)],
                dim=-1,
            )
        )

        image = inputs.image.reshape(-1, *self.observation_spaces.image.shape)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        image = self.image_net(image)
        image = self.image_linear(image)

        mission = inputs.mission.reshape(-1, *self.observation_spaces.mission.shape)
        mission = self.embed(mission.long())
        x = torch.cat([image, mission], dim=-1)

        assert self.is_recurrent
        x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs


class GRUEmbed(nn.Module):
    def __init__(self, num_embeddings: int):
        super().__init__()
        gru = nn.GRU(20, 64, batch_first=True)
        self.embed = nn.Sequential(
            nn.Embedding(num_embeddings, 20),
            gru,
        )

    def forward(self, x, **_):
        output, _ = self.embed.forward(x)
        return output[:, -1]
