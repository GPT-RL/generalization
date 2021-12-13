import functools
import os
import zipfile
from typing import Literal, cast, get_args

import main
import numpy as np
import pandas as pd
from envs import RenderWrapper, VecPyTorch
from mccrae.agent import Agent, ModelName
from mccrae.env import Env, RolloutsWrapper
from stable_baselines3.common.monitor import Monitor
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

BINARY_PRETRAINED = "binary-pretrained"
BINARY_UNTRAINED = "binary-untrained"
PRETRAINED = "pretrained"
UNTRAINED = "untrained"
BASELINE = "baseline"
# noinspection PyTypeHints
Architecture = Literal[
    BINARY_PRETRAINED, BINARY_UNTRAINED, PRETRAINED, UNTRAINED, BASELINE
]


class Args(main.Args):
    architecture: Architecture = BINARY_PRETRAINED
    config: str = None  # If given, yaml config from which to load params
    data_path: str = "mccrae.csv.zip"
    model_name: ModelName = "gpt2"
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    host_machine: str = os.getenv("HOST_MACHINE")
    n_features: int = 4
    n_train: int = 300
    room_size: int = 5
    train_ln: bool = False
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        main.configure_logger_args(self)


class ArgsType(main.ArgsType, Args):
    pass


class Trainer(main.Trainer):
    @classmethod
    def make_agent(cls, envs: VecPyTorch, args: ArgsType) -> Agent:
        observation_space, *_ = envs.get_attr("original_observation_space")
        return Agent(
            action_space=envs.action_space,
            model_name=args.model_name,
            hidden_size=args.hidden_size,
            observation_space=observation_space,
            recurrent=cls.recurrent(args),
        )

    @staticmethod
    def recurrent(args: Args):
        if "sequence" in args.env:
            assert args.recurrent
        return args.recurrent

    @classmethod
    def make_env(cls, env, allow_early_resets, render: bool = False, *args, **kwargs):
        def _thunk(
            concepts: np.ndarray,
            features: np.ndarray,
            max_token_id: int,
            room_size: int,
            seed: int,
            **_,
        ):
            _env = Env(
                concepts=concepts,
                features=features,
                max_token_id=max_token_id,
                room_size=room_size,
                seed=seed,
            )
            _env = RolloutsWrapper(_env)

            _env = Monitor(_env, allow_early_resets=allow_early_resets)
            if render:
                _env = RenderWrapper(_env)

            return _env

        return functools.partial(_thunk, **kwargs)

    @classmethod
    def make_vec_envs(cls, args: Args, device, test: bool, **kwargs):
        with zipfile.ZipFile(args.data_path) as zip_file:
            with zip_file.open("mccrae.csv") as file:
                df: pd.DataFrame = pd.read_csv(file, sep="\t")

        common_features = df.Feature.value_counts().index[: args.n_features]
        df = df[df.Feature.isin(common_features)]
        feature_arrays = df.groupby("Concept").apply(
            lambda group: (
                cast(
                    np.ndarray,
                    np.expand_dims(group.Feature.values, 0)
                    == np.expand_dims(common_features.values, 1),
                )
            ).any(1)
        )
        feature_arrays = feature_arrays[feature_arrays.apply(np.any)]
        features = np.stack(feature_arrays.values.tolist(), axis=0).astype(np.float32)

        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

        tokens = [
            tokenizer.encode(text, return_tensors="pt").squeeze(0)
            for text in feature_arrays.index
        ]
        concepts = pad_sequence(tokens, padding_value=tokenizer.eos_token_id).T.numpy()
        max_token_id = int(concepts.max()) + 1
        assert args.architecture in get_args(Architecture), args.architecture
        if args.architecture == BASELINE:
            _, unique = np.unique(concepts, return_inverse=True)
            concepts = unique.reshape(concepts.shape)

        idxs = np.arange(len(concepts))

        rng = np.random.default_rng(args.seed)
        rng.shuffle(idxs)

        train_idxs = idxs[: args.n_train]
        test_idxs = idxs[args.n_train :]
        concepts = concepts[test_idxs if test else train_idxs]
        features = features[test_idxs if test else train_idxs]

        return super().make_vec_envs(
            allow_early_resets=args.allow_early_resets,
            args=args,
            concepts=concepts,
            device=device,
            env=None,
            features=features,
            max_token_id=max_token_id,
            room_size=args.room_size,
            test=test,
            **kwargs,
        )


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
