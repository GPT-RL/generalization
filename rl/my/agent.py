from dataclasses import astuple

import agent
import numpy as np
import torch
import torch.nn as nn
from agent import NNBase
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from my.env import Obs
from transformers import CLIPModel
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
        clip: bool,
        gpt_embeddings: bool,
        hidden_size: int,
        image_size: int,
        observation_space: Dict,
        recurrent: bool,
        large_architecture: bool,
        train_ln: bool,
        train_wpe: bool,
    ):
        self.observation_spaces = Obs(**observation_space.spaces)

        if gpt_embeddings:
            *_, self.mission_size = self.observation_spaces.mission.shape
        else:
            self.mission_size = 2048
        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=image_size + self.mission_size,
            hidden_size=hidden_size,
        )
        self.clip = clip
        self.train_wpe = train_wpe
        self.train_ln = train_ln
        self.embeddings = None if gpt_embeddings else self.build_embeddings()

        image_shape = self.observation_spaces.image.shape
        d, h, w = image_shape

        if clip:
            self.clip: CLIPModel = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )

            for name, p in self.clip.vision_model.named_parameters():
                requires_grad = (self.train_wpe and "position_embedding" in name) or (
                    self.train_ln and "layer_norm" in name
                )
                p.requires_grad_(requires_grad)
        else:

            def get_image_net():
                prev = d
                if not large_architecture:
                    for (num_ch, kernel_size, stride) in [
                        (16, 8, 4),
                        (32, 4, 2),
                    ]:  # Downscale.
                        yield nn.Conv2d(
                            prev, num_ch, kernel_size=kernel_size, stride=stride
                        )
                        yield nn.ReLU()
                        prev = num_ch
                else:
                    for (num_ch, num_blocks) in [
                        (16, 2),
                        (32, 2),
                        (32, 2),
                    ]:  # Downscale.
                        yield nn.Conv2d(prev, num_ch, kernel_size=(3, 3), stride=(1, 1))
                        yield nn.MaxPool2d(
                            kernel_size=(3, 3),
                            stride=[2, 2],
                        )

                        # Residual block(s).
                        for _ in range(num_blocks):
                            yield ResidualBlock(num_ch)
                        prev = num_ch

                yield nn.ReLU()
                yield nn.Flatten()

            self._image_net = nn.Sequential(*get_image_net())

        dummy_input = torch.zeros(1, d, h, w)
        output = self.image_net(dummy_input)
        self.image_linear = nn.Sequential(
            nn.Linear(output.size(-1), image_size), nn.ReLU()
        )

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self.initial_hxs = nn.Parameter(self._initial_hxs)

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

    def image_net(self, image: torch.Tensor):
        if self.clip:
            state = self.clip.vision_model(pixel_values=image).last_hidden_state
            return state.mean(1)
        return self._image_net(image)

    def build_embeddings(self):
        num_embeddings = self.observation_spaces.mission.high.max()
        return nn.EmbeddingBag(int(num_embeddings) + 1, self.mission_size)

    def embed(self, inputs):
        if self.embeddings is not None:
            return self.embeddings.forward(inputs.long())
        return inputs

    def forward(self, inputs, rnn_hxs, masks):
        inputs = Obs(
            *torch.split(
                inputs,
                [get_size(space) for space in astuple(self.observation_spaces)],
                dim=-1,
            )
        )

        image = inputs.image.reshape(-1, *self.observation_spaces.image.shape)
        image = self.image_net(image)
        image = self.image_linear(image)

        mission = inputs.mission.reshape(-1, *self.observation_spaces.mission.shape)

        n, l, e = mission.shape
        flattened = mission.reshape(n * l, e)
        states = self.embed(flattened)
        states = states.reshape(n, l, -1)
        mission = states.mean(1)

        x = torch.cat([image, mission], dim=-1)

        assert self.is_recurrent
        x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs
