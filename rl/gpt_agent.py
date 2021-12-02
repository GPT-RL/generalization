import babyai_agent
import torch
from torch import nn
from utils import build_gpt


class Agent(babyai_agent.Agent):
    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class GPTEmbed(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        randomize_parameters: bool,
        train_wpe: bool,
        train_ln: bool,
    ):
        super().__init__()
        self.gpt = build_gpt(pretrained_model, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

    def forward(self, x, **_):
        return self.gpt.forward(x).last_hidden_state[:, -1]


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
        self.pretrained_model = pretrained_model
        self.randomize_parameters = randomize_parameters
        self.train_wpe = train_wpe
        self.train_ln = train_ln
        super().__init__(*args, pretrained_model=pretrained_model, **kwargs)

    def build_encodings(self, encoded):
        return nn.Embedding.from_pretrained(encoded.float())

    def embed_mission(self, mission: torch.Tensor):
        encoded = self.encodings.forward(mission.long())
        return (
            encoded
            if self.embeddings is None
            else self.embeddings.forward(encoded.long())
        )

    def build_embeddings(self):
        if self.train_wpe or self.train_ln:
            return GPTEmbed(
                pretrained_model=self.pretrained_model,
                randomize_parameters=self.randomize_parameters,
                train_wpe=self.train_wpe,
                train_ln=self.train_ln,
            )

    def embed(self, inputs):
        return self.embeddings.forward(inputs)
