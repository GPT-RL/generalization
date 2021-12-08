from __future__ import print_function

import logging
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional, cast, get_args

import numpy as np
import pandas as pd
import sweep_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from gql import gql
from run_logger import HasuraLogger
from spec import spec
from tap import Tap
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

GPTSize = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
BINARY_PRETRAINED = "binary-pretrained"
BINARY_UNTRAINED = "binary-untrained"
PRETRAINED = "pretrained"
UNTRAINED = "untrained"
BASELINE = "baseline"
# noinspection PyTypeHints
Architecture = Literal[
    BINARY_PRETRAINED, BINARY_UNTRAINED, PRETRAINED, UNTRAINED, BASELINE
]


def build_gpt(gpt_size: GPTSize, randomize_parameters: bool):
    return (
        GPT2Model(
            GPT2Config.from_pretrained(
                gpt_size,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        )
        if randomize_parameters
        else GPT2Model.from_pretrained(
            gpt_size,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    )


class GPTEmbed(nn.Module):
    def __init__(
        self,
        model_name: GPTSize,
        randomize_parameters: bool,
        train_wpe: bool,
        train_ln: bool,
    ):
        super().__init__()
        self.gpt = build_gpt(model_name, randomize_parameters)
        for name, p in self.gpt.named_parameters():
            requires_grad = (train_wpe and "wpe" in name) or (train_ln and "ln" in name)
            p.requires_grad_(requires_grad)

    def forward(self, x, **_):
        return self.gpt.forward(x).last_hidden_state[:, -1]


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def f(x):
    breakpoint()
    return x


class Net(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        encoder: nn.Module,
        hidden_size: int,
        output_size: int,
    ):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            encoder,
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def shuffle(df: pd.DataFrame, **kwargs):
    return df.sample(frac=1, **kwargs).reset_index(drop=True)


NON_ANTONYM = "non-antonyms"
ANTONYM = "antonyms"
LEMMA = "lemma"
TARGET = "target"


@dataclass
class _Dataset(Dataset):
    inputs: np.ndarray
    targets: np.ndarray

    def __post_init__(self):
        assert len(self.inputs) == len(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) -> T_co:
        return self.inputs[index], self.targets[index]


RUN_OR_SWEEP = Literal["run", "sweep"]


class Run(Tap):
    name: str

    def configure(self) -> None:
        self.add_argument("name", type=str)  # positional


class Sweep(Tap):
    sweep_id: int = None


def configure_logger_args(args: Tap):
    args.add_subparser("run", Run)
    args.add_subparser("sweep", Sweep)


class Args(Tap):
    architecture: Architecture = BINARY_PRETRAINED
    batch_size: int = 32
    config: Optional[str] = None  # If given, yaml config from which to load params
    data_path: str = "mccrae.csv.zip"
    dry_run: bool = False
    model_name: GPTSize = "gpt2"
    epochs: int = 14
    gamma: float = 0.99
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    hidden_size: int = 1024
    host_machine: str = os.getenv("HOST_MACHINE")
    load_id: int = None  # path to load parameters from if at all
    log_interval: int = 10
    log_level: str = "INFO"
    lr: float = 1.0
    n_features: int = 4
    n_train: int = 300
    no_cuda: bool = False
    save_interval: int = None
    seed: int = 1
    test_batch_size: int = 1000
    train_ln: bool = False
    train_wpe: bool = False

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        self.add_subparser("run", Run)
        self.add_subparser("sweep", Sweep)


class ArgsType(Args):
    logger_args: Optional[RUN_OR_SWEEP]


def get_save_path(run_id: Optional[int]):
    return (
        Path("/tmp/logs/checkpoint.pkl")
        if run_id is None
        else Path("/tmp/logs", str(run_id), "checkpoint.pkl")
    )


def train(args: Args, logger: HasuraLogger):
    pprint(args.as_dict())
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = dict(batch_size=args.batch_size)
    test_kwargs = dict(batch_size=args.test_batch_size)
    if use_cuda:
        cuda_kwargs = dict(num_workers=1, shuffle=True, pin_memory=True)
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

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
    targets = np.stack(feature_arrays.values.tolist(), axis=0).astype(np.float32)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    tokens = [
        tokenizer.encode(text, return_tensors="pt").squeeze(0)
        for text in feature_arrays.index
    ]
    inputs = pad_sequence(tokens, padding_value=tokenizer.eos_token_id).T.numpy()
    assert args.architecture in get_args(Architecture), args.architecture
    if args.architecture == BASELINE:
        _, unique = np.unique(inputs, return_inverse=True)
        inputs = unique.reshape(inputs.shape)

    idxs = np.arange(len(inputs))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(idxs)
    train_idxs = idxs[: args.n_train]
    test_idxs = idxs[args.n_train :]

    train_inputs = inputs[train_idxs]
    train_targets = targets[train_idxs]

    embedding_size = GPT2Config.from_pretrained(args.model_name).n_embd
    encoder = (
        nn.EmbeddingBag(int(inputs.max()), embedding_size)
        if args.architecture == BASELINE
        else GPTEmbed(
            model_name=args.model_name,
            randomize_parameters=args.architecture == UNTRAINED,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
        )
    ).to(device)
    with torch.no_grad():
        outputs = encoder(torch.tensor(train_inputs).to(device))
    mean = outputs.mean(dim=0, keepdims=True)
    std = outputs.mean(dim=0, keepdims=True)

    prior = torch.tensor(train_targets.mean(axis=0, keepdims=True)).to(device)
    train_dataset = _Dataset(inputs=train_inputs, targets=train_targets)
    test_dataset = _Dataset(inputs=inputs[test_idxs], targets=targets[test_idxs])
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    if args.architecture in [BINARY_PRETRAINED, BINARY_UNTRAINED]:
        encoder = nn.Sequential(
            encoder,
            Lambda(lambda x: (x - mean) / std),
            nn.Sigmoid(),
            *([] if args.train_ln or args.train_wpe else [Lambda(lambda x: x.round())]),
        )

    model = Net(
        embedding_size=embedding_size,
        encoder=encoder,
        hidden_size=args.hidden_size,
        output_size=targets.shape[1],
    ).to(device)
    start_time = time.time()

    def evaluate(_epoch):
        test_loss = 0
        _accurate = []
        _recalled = []
        with torch.no_grad():
            for _data, _target in test_loader:
                _data, _target = _data.to(device), _target.to(device)
                _output = model(_data)
                test_loss += F.binary_cross_entropy(
                    _output, _target, reduction="sum"
                ).item()  # sum up batch loss
                _pred = _output.round()
                _correct = _pred.eq(_target.view_as(_pred)).squeeze(-1).float()
                _accurate += [_correct]
                _recalled += [_correct[_target.bool()]]
        test_loss /= len(test_loader.dataset)
        test_accuracy = torch.cat(_accurate).mean()
        test_recall = torch.cat(_recalled).mean()
        _log = {
            EPOCH: _epoch,
            TEST_LOSS: test_loss,
            TEST_ACCURACY: test_accuracy.item(),
            TEST_RECALL: test_recall.item(),
            RUN_ID: logger.run_id,
            HOURS: (time.time() - start_time) / 3600,
        }
        pprint(_log)
        if logger.run_id is not None:
            logger.log(_log)

    save_path = get_save_path(logger.run_id)
    if args.load_id is not None:
        load_path = get_save_path(args.load_id)
        logging.info(f"Loading checkpoint from {load_path}...")
        model.load_state_dict(torch.load(load_path))
    if args.save_interval:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    save_count = 0
    frames = 0
    for epoch in range(args.epochs):

        correct = []
        for batch_idx, (data, target) in enumerate(train_loader):
            if (
                batch_idx == 0
                and args.save_interval is not None
                and epoch % args.save_interval == 0
            ):
                torch.save(model.state_dict(), str(save_path))
                save_count += 1

            if batch_idx == 0 and epoch % args.log_interval == 0:
                evaluate(_epoch=epoch)

            frames += len(data)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            weight = target / prior + (1 - target) / (1 - prior)
            loss = F.binary_cross_entropy(output, target, weight=weight / 2)
            pred = output.round()
            correct += [pred.eq(target.view_as(pred)).squeeze(-1).float()]
            if batch_idx == 0 and epoch % args.log_interval == 0:
                accuracy = torch.cat(correct).mean()
                seconds = time.time() - start_time
                log = {
                    EPOCH: epoch,
                    ACCURACY: accuracy.item(),
                    RUN_ID: logger.run_id,
                    HOURS: seconds / 3600,
                    SAVE_COUNT: save_count,
                    FPS: frames / seconds,
                    LOSS: loss.item(),
                }
                pprint(log)
                if logger.run_id is not None:
                    logger.log(log)

                if args.dry_run:
                    break
            loss.backward()
            optimizer.step()

        scheduler.step()


EXCLUDED = {
    "config",
    "name",
    "sync_envs",
    "render",
    "render_test",
    "subcommand",
    "sweep_id",
    "load_id",
    "logger_args",
}

FPS = "fps"
GRADIENT_NORM = "gradient norm"
TIME = "time"
HOURS = "hours"
EPOCH = "epoch"
SAVE_COUNT = "save count"
LOSS = "loss"
TEST_LOSS = "test loss"
ACCURACY = "accuracy"
RECALL = "recall"
TEST_ACCURACY = "test accuracy"
TEST_RECALL = "test recall"
RUN_ID = "run ID"


def update_args(args, parameters, check_hasattr=True):
    for k, v in parameters.items():
        if k not in EXCLUDED:
            if check_hasattr:
                assert hasattr(args, k), k
            setattr(args, k, v)


def main(args: ArgsType):
    logging.getLogger().setLevel(args.log_level)
    if args.config is not None:
        with Path(args.config).open() as f:
            config = yaml.load(f, yaml.FullLoader)
            args = args.from_dict(
                {k: v for k, v in config.items() if k not in EXCLUDED}
            )

    metadata = dict(reproducibility_info=args.get_reproducibility_info())
    if args.host_machine:
        metadata.update(host_machine=args.host_machine)
    if name := getattr(args, "name", None):
        metadata.update(name=name)

    logger: HasuraLogger
    with HasuraLogger(args.graphql_endpoint) as logger:
        valid = (*get_args(RUN_OR_SWEEP), None)
        assert args.logger_args in valid, f"{args.logger_args} is not in {valid}."

        if args.logger_args is not None:
            charts = [spec(x=HOURS, y=y) for y in (ACCURACY, TEST_ACCURACY,)] + [
                spec(x=EPOCH, y=y)
                for y in (
                    SAVE_COUNT,
                    FPS,
                    LOSS,
                    ACCURACY,
                    TEST_LOSS,
                    TEST_ACCURACY,
                    TEST_RECALL,
                )
            ]
            metadata = dict(reproducibility_info=args.get_reproducibility_info())
            if args.host_machine:
                metadata.update(host_machine=args.host_machine)
            if name := getattr(args, "name", None):
                metadata.update(name=name)

            params, logger = sweep_logger.initialize(
                graphql_endpoint=args.graphql_endpoint,
                config=args.config,
                charts=charts,
                sweep_id=getattr(args, "sweep_id", None),
                load_id=args.load_id,
                create_run=args.logger_args is not None,
                params=args.as_dict(),
                metadata=metadata,
            )

            update_args(args, params)

        if args.load_id is not None:
            parameters = logger.execute(
                gql(
                    """
query GetParameters($id: Int!) {
  run_by_pk(id: $id) {
    metadata(path: "parameters")
  }
}"""
                ),
                variable_values=dict(id=args.load_id),
            )["run_by_pk"]["metadata"]
            update_args(args, parameters, check_hasattr=False)
        return train(args=args, logger=logger)


if __name__ == "__main__":
    main(cast(ArgsType, Args().parse_args()))
