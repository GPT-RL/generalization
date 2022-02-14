from dataclasses import astuple

import agent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import NNBase
from gym import Space
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from my.env import Obs
from transformers import BertConfig, GPT2Config, GPT2Tokenizer, GPTNeoConfig


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


class Lambda(nn.Module):
    def __init__(self, f: callable):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=imm_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels,
            out_channels=out_features,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
    def __init__(self, max_value, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3 * max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(
            inputs.device
        )
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class Base(NNBase):
    def __init__(
        self,
        attn_temp: float,
        device: torch.DeviceObjType,
        freeze_keys: bool,
        hidden_size: int,
        missions: torch.Tensor,
        qkv: bool,
        observation_space: Dict,
        pretrained_model: str,
        recurrent: bool,
    ):
        super().__init__(
            recurrent=recurrent,
            recurrent_input_size=128,
            hidden_size=128,
        )
        self.observation_spaces = Obs(*observation_space.spaces)
        self.num_directions = self.observation_spaces.direction.n
        self.num_actions = self.observation_spaces.action.n
        self.qkv = qkv

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

        image_dim = 128
        memory_dim = 128
        instr_dim = 128
        lang_model = "gru"
        use_memory = False
        arch = "bow_endpool_res"
        aux_info = None
        endpool = "endpool" in arch
        use_bow = "bow" in arch
        pixel = "pixel" in arch
        self.res = "res" in arch

        # Decide which components are enabled
        self.use_instr = True
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        for part in self.arch.split("_"):
            if part not in ["original", "bow", "pixels", "endpool", "res"]:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        # if not self.use_instr:
        #     raise ValueError("FiLM architecture can be used when instructions are enabled")
        self.image_conv = nn.Sequential(
            *[
                *(
                    [
                        ImageBOWEmbedding(
                            int(self.observation_spaces.image.high.max()), 128
                        )
                    ]
                    if use_bow
                    else []
                ),
                *(
                    [
                        nn.Conv2d(
                            in_channels=3,
                            out_channels=128,
                            kernel_size=(8, 8),
                            stride=8,
                            padding=0,
                        )
                    ]
                    if pixel
                    else []
                ),
                nn.Conv2d(
                    in_channels=128 if use_bow or pixel else 3,
                    out_channels=128,
                    kernel_size=(3, 3) if endpool else (2, 2),
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
                nn.Conv2d(
                    in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            ]
        )
        self.film_pool = nn.MaxPool2d(
            kernel_size=self.observation_spaces.image.shape[-2:] if endpool else (2, 2),
            stride=2,
        )

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ["gru", "bigru", "attgru"]:
                self.word_embedding = nn.Embedding(
                    int(self.observation_spaces.mission.nvec.max()), self.instr_dim
                )
                if self.lang_model in ["gru", "bigru", "attgru"]:
                    gru_dim = self.instr_dim
                    if self.lang_model in ["bigru", "attgru"]:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim,
                        gru_dim,
                        batch_first=True,
                        bidirectional=(self.lang_model in ["bigru", "attgru"]),
                    )
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList(
                        [
                            nn.Conv2d(1, kernel_dim, (K, self.instr_dim))
                            for K in kernel_sizes
                        ]
                    )
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == "attgru":
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            num_module = 2
            self.controllers = []
            for ni in range(num_module):
                mod = FiLM(
                    in_features=self.final_instr_dim,
                    out_features=128 if ni < num_module - 1 else self.image_dim,
                    in_channels=128,
                    imm_channels=128,
                )
                self.controllers.append(mod)
                self.add_module("FiLM_" + str(ni), mod)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, inputs, rnn_hxs, masks):
        obs = Obs(
            *torch.split(
                inputs,
                [get_size(space) for space in astuple(self.observation_spaces)],
                dim=-1,
            )
        )

        instr_embedding = self._get_instr_embedding(obs.mission.long())

        image = obs.image.reshape(-1, *self.observation_spaces.image.shape)
        x = torch.transpose(torch.transpose(image, 1, 3), 2, 3)

        if "pixel" in self.arch:
            x /= 256.0
        x = self.image_conv(x)
        if self.use_instr:
            for controller in self.controllers:
                out = controller(x, instr_embedding)
                if self.res:
                    out += x
                x = out
        x = x.max(-1).values.max(-1).values
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        return self.critic(x), x, rnn_hxs

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == "gru":
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths - 1, :]
            return hidden
