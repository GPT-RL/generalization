from my import agent


class Agent(agent.Agent):
    def build_base(self, obs_shape, **kwargs):
        return Base(**kwargs)


class Base(agent.Base):
    def build_embeddings(self):
        return None

    def embed(self, inputs):
        return inputs
