import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import pybullet_agent
import torch
from multihead_attention import MultiheadAttention
from torch import Tensor, nn
from torch.nn import Parameter
from transformers import BertConfig, GPT2Config, GPTNeoConfig
from utils import build_gpt


class Agent(pybullet_agent.Agent):
    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


def get_primes():
    for i in itertools.count():
        if not any(i % j == 0 for j in range(2, i // 2 + 1)):
            yield i


@lru_cache()
def get_primes_tensor(num_el, device, shape):
    return torch.tensor(
        list(itertools.islice(get_primes(), num_el)),
        device=device,
    ).reshape(shape)


@dataclass(frozen=True)
class HashTensorWrapper:
    """
    https://discuss.pytorch.org/t/how-to-put-tensors-in-a-set/123836/7
    """

    tensor: Tensor

    def __hash__(self):
        primes = get_primes_tensor(
            self.tensor.numel(), self.tensor.device, self.tensor.shape
        )
        return torch.sum(self.tensor * primes).item()

    def __eq__(self, other):
        assert isinstance(other, HashTensorWrapper)
        equals = self.tensor == other.tensor
        equals = cast(Tensor, equals)
        return torch.all(equals)


class Base(pybullet_agent.Base):
    def __init__(
        self,
        *args,
        attn_temp: float,
        device: torch.DeviceObjType,
        freeze_keys: bool,
        multihead_attention: bool,
        missions: torch.Tensor,
        pretrained_model: str,
        randomize_parameters: bool,
        train_ln: bool,
        train_wpe: bool,
        **kwargs,
    ):
        self.multihead_attention = multihead_attention
        self._embedding_size = pretrained_model
        self.randomize_parameters = randomize_parameters
        self.train_wpe = train_wpe
        self.train_ln = train_ln

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

        super().__init__(
            *args,
            mission_size=config.n_embd,
            pretrained_model=pretrained_model,
            **kwargs,
        )
        if multihead_attention:
            self.embeddings.to(device)
            missions = missions.to(device)
            outputs = self.gpt_forward_pass(missions)
            outputs = outputs.reshape(-1, outputs.size(-1))
            self.keys = Parameter(attn_temp * outputs, requires_grad=not freeze_keys)
            self.values = nn.Embedding(*outputs.shape)
            self.multihead_attn = MultiheadAttention(self.embedding_size, num_heads=1)

    def build_embeddings(self):
        gpt = build_gpt(self._embedding_size, self.randomize_parameters)
        for name, p in gpt.named_parameters():
            requires_grad = (self.train_wpe and "wpe" in name) or (
                self.train_ln and "ln" in name
            )
            p.requires_grad_(requires_grad)
        return gpt

    @lru_cache()
    def cached_gpt_forward_pass(self, inputs: HashTensorWrapper):
        with torch.no_grad():
            return self.uncached_gpt_forward_pass(inputs.tensor)

    def uncached_gpt_forward_pass(self, tensor):
        return self.embeddings.forward(
            tensor, attention_mask=tensor != self.pad_token_id
        ).last_hidden_state

    def gpt_forward_pass(self, inputs):
        if self.train_ln or self.train_wpe:
            return self.uncached_gpt_forward_pass(inputs)
        else:
            return torch.cat(
                [
                    self.cached_gpt_forward_pass(HashTensorWrapper(x))
                    for x in inputs.unsqueeze(1)
                ],
                dim=0,
            )

    def embed(self, inputs):
        # inputs = inputs.reshape(-1, *self.observation_spaces.mission.nvec.shape)
        # n, l, e = inputs.shape
        # flattened = inputs.reshape(n * l, e)
        states = self.gpt_forward_pass(inputs)
        # states = states.mean(1).reshape(n, l, -1)
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
            return attn_output.mean(0)
        else:
            return states.mean(1)
