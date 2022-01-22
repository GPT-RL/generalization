import babyai_agent
from torch import nn
from utils import build_gpt


class Agent(babyai_agent.Agent):
    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class Base(babyai_agent.Base):
    def __init__(
        self,
        *args,
        num_embeddings: int,
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
        self.keys = nn.Embedding(num_embeddings, self.embedding_size)
        self.values = nn.Embedding(num_embeddings, self.embedding_size)
        self.multihead_attn = nn.MultiheadAttention(self.embedding_size, num_heads=1)

    def build_embeddings(self):
        gpt = build_gpt(self._embedding_size, self.randomize_parameters)
        for name, p in gpt.named_parameters():
            requires_grad = (self.train_wpe and "wpe" in name) or (
                self.train_ln and "ln" in name
            )
            p.requires_grad_(requires_grad)
        return gpt

    def embed(self, inputs):
        states = self.embeddings.forward(
            inputs, attention_mask=inputs != self.pad_token_id
        ).last_hidden_state[:, -2:]
        key = self.keys.weight.unsqueeze(1)
        value = self.values.weight.unsqueeze(1)
        attn_output, _ = self.multihead_attn.forward(query=states, key=key, value=value)
        return attn_output.mean(1)
