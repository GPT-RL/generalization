import itertools
from functools import lru_cache

import torch
from my import agent
from transformers import GPT2Config
from wrappers import GPT3Tokenizer


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
        gpt_embeddings: bool,
        randomize_parameters: bool,
        **kwargs,
    ):
        self.gpt_embeddings = gpt_embeddings
        self.randomize_parameters = randomize_parameters
        self.train_wpe = False  # todo
        self.train_ln = False  # todo

        if gpt_embeddings:
            n_embed = GPT3Tokenizer().n_embed
        else:
            config = GPT2Config.from_pretrained(
                "gpt2",
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            n_embed = config.n_embd

        super().__init__(
            *args,
            mission_size=n_embed,
            gpt_embeddings=gpt_embeddings,
            **kwargs,
        )

    def build_embeddings(self):
        return None

    def embed(self, inputs):
        return inputs
