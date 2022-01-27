import babyai_agent
import torch
from multihead_attention import MultiheadAttention
from torch import nn
from torch.nn import Parameter
from utils import build_gpt


class Agent(babyai_agent.Agent):
    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class Base(babyai_agent.Base):
    def __init__(
        self,
        *args,
        attn_temp: float,
        freeze_keys: bool,
        gpt: nn.Module,
        multihead_attention: bool,
        outputs: torch.Tensor,
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
        super().__init__(*args, pretrained_model=pretrained_model, **kwargs)
        self.gpt = gpt
        if multihead_attention:
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

    def embed(self, inputs):
        embeddings = self.embeddings
        pad_token_id = self.pad_token_id
        states = self.pass_through_gpt(embeddings, inputs, pad_token_id)
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

    @staticmethod
    def pass_through_gpt(gpt, inputs, pad_token_id):
        return gpt.forward(
            inputs, attention_mask=inputs != pad_token_id
        ).last_hidden_state[:, -2:]
