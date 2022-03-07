import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import torch
from my import agent
from torch import Tensor
from transformers import BertConfig, GPT2Config, GPTNeoConfig
from utils import build_gpt


class Agent(agent.Agent):
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


class Base(agent.Base):
    def __init__(
        self,
        *args,
        pretrained_model: str,
        randomize_parameters: bool,
        **kwargs,
    ):
        self._embedding_size = pretrained_model
        self.randomize_parameters = randomize_parameters
        self.train_wpe = False  # todo
        self.train_ln = False  # todo

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

    def build_embeddings(self):
        gpt = build_gpt(self._embedding_size, self.randomize_parameters)
        for name, p in gpt.named_parameters():
            requires_grad = (self.train_wpe and "wpe" in name) or (
                self.train_ln and "ln" in name
            )
            p.requires_grad_(requires_grad)
        return gpt

    @lru_cache()
    def cached_gpt_forward_pass(self, inputs: "HashTensorWrapper"):
        with torch.no_grad():
            return self.uncached_gpt_forward_pass(inputs.tensor)

    def embed(self, inputs):
        states = self.gpt_forward_pass(inputs)
        return states.mean(
            1
        )  # mean across input length  (if there are 3 tokens, gpt will output an (n, 3, d) size embedding).

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

    def uncached_gpt_forward_pass(self, tensor):
        return self.embeddings.forward(
            tensor, attention_mask=tensor != self.pad_token_id
        ).last_hidden_state


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
