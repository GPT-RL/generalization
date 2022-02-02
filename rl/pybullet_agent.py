from dataclasses import astuple

import agent
import numpy as np
import torch
import torch.nn as nn
from agent import NNBase
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from pybullet_env import Observation
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
        spaces = Observation(*observation_space.spaces)
        super().__init__(
            obs_shape=spaces.image.shape, observation_space=observation_space, **kwargs
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


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
        self.observation_spaces = Observation(*observation_space.spaces)

        self.pad_token_id = GPT2Tokenizer.from_pretrained(pretrained_model).eos_token_id

        self.embeddings = self.build_embeddings()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        image_shape = self.observation_spaces.image.shape
        h, w, d = image_shape
        dummy_input = torch.zeros(1, d, h, w)

        self.image_conv = nn.Sequential(
            init_(nn.Conv2d(d, 16, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        output = self.image_conv(dummy_input)
        self.image_linear = nn.Linear(output.size(-1), 256)

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
        return self.embeddings.forward(inputs.mission.long())

    def forward(self, inputs, rnn_hxs, masks):
        inputs = Observation(
            *torch.split(
                inputs,
                [get_size(space) for space in astuple(self.observation_spaces)],
                dim=-1,
            )
        )

        image = inputs.image.reshape(-1, *self.observation_spaces.image.shape)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        image = self.image_conv(image)
        image = self.image_linear(image)

        mission = self.embed(inputs.mission.long())
        x = torch.cat([image, mission], dim=-1)

        assert self.is_recurrent
        x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs
