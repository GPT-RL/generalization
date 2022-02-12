from dataclasses import astuple

import agent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import NNBase
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from multihead_attention import MultiheadAttention
from my.env import Obs
from torch.nn import Parameter
from transformers import BertConfig, GPT2Config, GPT2Tokenizer, GPTNeoConfig
from utils import init


def get_size(space: Space):
    if isinstance(space, (Box, MultiDiscrete)):
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return 1
    raise TypeError()


class Agent(agent.Agent):
    def __init__(self, observation_space, **kwargs):
        spaces = Obs(*observation_space.spaces)
        super().__init__(
            obs_shape=spaces.image.shape, observation_space=observation_space, **kwargs
        )

    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class GRUEmbed(nn.Module):
    def __init__(self, num_embeddings: int, hidden_size: int, output_size: int):
        super().__init__()
        gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.embed = nn.Sequential(
            nn.Embedding(num_embeddings, hidden_size),
            gru,
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, x, **_):
        hidden = self.embed.forward(x)[1].squeeze(0)
        return self.projection(hidden)


class Base(NNBase):
    def __init__(
        self,
        attn_temp: float,
        device: torch.DeviceObjType,
        freeze_keys: bool,
        hidden_size: int,
        missions: torch.Tensor,
        multihead_attention: bool,
        observation_space: Dict,
        pretrained_model: str,
        recurrent: bool,
    ):
        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.observation_spaces = Obs(*observation_space.spaces)
        self.num_directions = self.observation_spaces.direction.n
        self.num_actions = self.observation_spaces.action.n
        self.multihead_attention = multihead_attention

        self.pad_token_id = GPT2Tokenizer.from_pretrained(pretrained_model).eos_token_id

        if "neo" in pretrained_model:
            config = GPTNeoConfig.from_pretrained(
                pretrained_model,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            self.embedding_size = config.hidden_size
        elif "gpt2" in pretrained_model:
            config = GPT2Config.from_pretrained(
                pretrained_model,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            self.embedding_size = config.n_embd
        elif "bert" in pretrained_model:
            config = BertConfig.from_pretrained(
                pretrained_model,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            self.embedding_size = config.hidden_size

        else:
            raise RuntimeError(f"Invalid model name: {pretrained_model}")

        self.embeddings = self.build_embeddings()

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

            # self.image_net = nn.Sequential(
            #     init_(nn.Conv2d(d, 32, 8, stride=4)),
            #     nn.ReLU(),
            #     init_(nn.Conv2d(32, 64, 4, stride=2)),
            #     nn.ReLU(),
            #     init_(nn.Conv2d(64, 32, 3, stride=1)),
            #     nn.ReLU(),
            #     nn.Flatten(),
            # )
            # try:
            #     output = self.image_net(dummy_input)
            # except RuntimeError:
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
                    output.size(-1)
                    + self.num_directions
                    + self.num_actions
                    + self.embedding_size,
                    hidden_size,
                )
            ),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        if multihead_attention:
            self.embeddings.to(device)
            missions = missions.to(device)
            outputs = self.embed(missions)
            outputs = outputs.reshape(-1, outputs.size(-1))
            self.keys = Parameter(attn_temp * outputs, requires_grad=not freeze_keys)
            self.values = nn.Embedding(*outputs.shape)
            self.multihead_attn = MultiheadAttention(self.embedding_size, num_heads=1)

    def build_embeddings(self):
        num_embeddings = int(self.observation_spaces.mission.nvec.flatten()[0])
        return nn.Embedding(
            num_embeddings, self.embedding_size, padding_idx=self.pad_token_id
        )

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
        directions = inputs.direction.long()
        directions = F.one_hot(directions, num_classes=self.num_directions).squeeze(1)
        action = inputs.action.long()
        action = F.one_hot(action, num_classes=self.num_actions).squeeze(1)

        mission = inputs.mission.reshape(-1, *self.observation_spaces.mission.shape)

        n, l, e = mission.shape
        flattened = mission.reshape(n * l, e)
        states = self.embed(flattened.long())
        states = states.mean(1).reshape(n, l, -1)
        if self.multihead_attention:
            query = states.transpose(0, 1)
            n = query.size(1)
            key = self.keys.unsqueeze(1).repeat(1, n, 1)
            value = self.values.weight.unsqueeze(1).repeat(1, n, 1)
            attn_output, _ = self.multihead_attn.forward(
                query=query, key=key, value=value
            )
            # print((100 * _.max(dim=-1).values).round())
            # breakpoint()
            mission = attn_output.mean(0)
        else:
            mission = states.mean(1)

        x = torch.cat([image, directions, action, mission], dim=-1)
        x = self.merge(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs

    def embed(self, inputs):
        return self.embeddings.forward(inputs)
