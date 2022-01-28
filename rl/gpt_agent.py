import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import babyai_agent
import torch
from torch import Tensor
from utils import build_gpt


class Agent(babyai_agent.Agent):
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


class Base(babyai_agent.Base):
    def __init__(
        self,
        *args,
        pretrained_model: str,
        randomize_parameters: bool,
        train_ln: bool,
        train_wpe: bool,
        **kwargs,
    ):
        self._embedding_size = pretrained_model
        self.randomize_parameters = randomize_parameters
        self.train_wpe = train_wpe
        self.train_ln = train_ln
        super().__init__(*args, pretrained_model=pretrained_model, **kwargs)

    def build_embeddings(self):
        gpt = build_gpt(self._embedding_size, self.randomize_parameters)
        for name, p in gpt.named_parameters():
            requires_grad = (self.train_wpe and "wpe" in name) or (
                self.train_ln and "ln" in name
            )
            p.requires_grad_(requires_grad)
        return gpt

    @staticmethod
    def process_last_hidden_state(last_hidden_state: Tensor):
        return last_hidden_state.mean(1)

    @lru_cache()
    def _embed(self, inputs: HashTensorWrapper):
        with torch.no_grad():
            forward = self.embeddings.forward(
                inputs.tensor, attention_mask=inputs.tensor != self.pad_token_id
            )
        return self.process_last_hidden_state(forward.last_hidden_state)

    def embed(self, inputs):
        if self.train_ln or self.train_wpe:
            state = self.embeddings.forward(
                inputs, attention_mask=inputs != self.pad_token_id
            ).last_hidden_state
            return self.process_last_hidden_state(state)
        else:
            embedded = [self._embed(HashTensorWrapper(x)) for x in inputs.unsqueeze(1)]
            return torch.cat(embedded, dim=0)
