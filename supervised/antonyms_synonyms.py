from __future__ import print_function

import logging
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import List, Literal, Optional, cast, get_args

import kaggle
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
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

ModelName = Literal["gpt-2", "gpt3-medium", "gpt2-large", "gpt2-xl"]


def build_gpt(gpt_size: ModelName, randomize_parameters: bool):
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
        model_name: ModelName,
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
        return self.gpt.forward(x).last_hidden_state[:, :, -1]


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Net(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        encoder: nn.Module,
        hidden_size: int,
        multiplicative_interaction: bool,
    ):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            encoder,
            nn.Linear(embedding_size, hidden_size),
            Lambda(lambda x: x.prod(1))
            if multiplicative_interaction
            else nn.Sequential(Lambda(lambda x: x.reshape(x.size(0), -1))),
            nn.ReLU(),
            nn.Linear((1 if multiplicative_interaction else 2) * hidden_size, 1),
            Lambda(lambda x: x.squeeze(-1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


LEMMA = "lemma"
ANTONYMS = "antonyms"
SYNONYMS = "synonyms"
COUNTERPART = "counterpart"
TARGET = "target"


def explode(data: pd.DataFrame, column):
    data: pd.DataFrame = data.dropna(subset=[LEMMA, column])
    split = data[column].str.split(pat="[;|]")
    data = data.assign(**{column: split})
    data = data.explode(column)
    return data


def get_tensors(
    data: pd.DataFrame, column: str, tokenizer: GPT2Tokenizer
) -> torch.Tensor:
    data: pd.DataFrame = data.dropna(subset=[LEMMA, column])
    split = data[column].str.split(pat="[;|]")
    data = data.assign(**{column: split})
    data = data.explode(column)
    data = data[[LEMMA, column]]

    for col in [LEMMA, column]:
        with tqdm(desc=f"Encoding {col}", total=len(data)) as pbar:

            def encode(s: str):
                pbar.update(1)
                tensor = tokenizer.encode(s, return_tensors="pt")
                tensor = cast(torch.Tensor, tensor)
                return tensor.T

            encoded = data[col].apply(encode)

        yield from encoded.values.tolist()


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


ModelName = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
BINARY_PRETRAINED = "binary-pretrained"
BINARY_UNTRAINED = "binary-untrained"
PRETRAINED = "pretrained"
UNTRAINED = "untrained"
BASELINE = "baseline"
# noinspection PyTypeHints
Architecture = Literal[
    BINARY_PRETRAINED, BINARY_UNTRAINED, PRETRAINED, UNTRAINED, BASELINE
]


class Args(Tap):
    antonyms_dir: str = "/root/.cache/data/antonyms.csv"
    architecture: Architecture = BINARY_PRETRAINED
    batch_size: int = 32
    config: Optional[str] = None  # If given, yaml config from which to load params
    dry_run: bool = False
    model_name: ModelName = "gpt2-large"
    epochs: int = 14
    gamma: float = 0.99
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    hidden_size: int = 512
    host_machine: str = os.getenv("HOST_MACHINE")
    load_id: int = None  # path to load parameters from if at all
    log_interval: int = 10
    log_level: str = "INFO"
    lr: float = 1.0
    multiplicative_interaction: bool = False
    n_classes: int = 3
    n_train: int = 9000
    n_test: int = 320
    no_cuda: bool = False
    save_interval: int = None
    seed: int = 1
    synonyms_dir: str = "/root/.cache/data/synonyms.csv"
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


def shuffle(tensor: torch.Tensor):
    idxs = torch.randperm(len(tensor))
    return tensor[idxs]


def create_dataset(inputs, targets, in_dataset):
    inputs = inputs[in_dataset]
    targets = targets[in_dataset]
    idxs = torch.randperm(in_dataset.sum())
    inputs = inputs[idxs]
    targets = targets[idxs]
    return _Dataset(inputs, targets)


def train(args: Args, logger: HasuraLogger):
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    kaggle.api.authenticate()

    antonyms_path = Path(args.antonyms_dir, "antonyms-wordnet.zip")
    if not antonyms_path.exists():
        kaggle.api.dataset_download_files(
            "duketemon/antonyms-wordnet",
            path=args.antonyms_dir,
            unzip=False,
        )
    with zipfile.ZipFile(str(antonyms_path), "r") as zip_ref:
        with zip_ref.open("antonyms.csv") as f:
            antonyms = pd.read_csv(f)

    synonyms_path = Path(args.synonyms_dir, "wordnet-synonyms.zip")
    if not synonyms_path.exists():
        kaggle.api.dataset_download_files(
            "duketemon/wordnet-synonyms",
            path=args.synonyms_dir,
            unzip=False,
        )
    with zipfile.ZipFile(str(synonyms_path), "r") as zip_ref:
        with zip_ref.open("synonyms.csv") as f:
            synonyms = pd.read_csv(f)

    antonyms: List[torch.Tensor] = list(get_tensors(antonyms, ANTONYMS, tokenizer))
    synonyms: List[torch.Tensor] = list(get_tensors(synonyms, SYNONYMS, tokenizer))
    ns = len(synonyms) // 2
    na = len(antonyms) // 2
    assert ns > na
    inputs: torch.Tensor = (
        pad_sequence(
            [*antonyms, *synonyms],
            padding_value=tokenizer.eos_token_id,
        )
        .squeeze(-1)
        .T
    )
    d = inputs.shape[-1]
    antonyms = inputs[: 2 * na].reshape(2, na, d).swapaxes(0, 1)
    synonyms = inputs[2 * na :].reshape(2, ns, d).swapaxes(0, 1)
    antonyms: torch.Tensor = shuffle(cast(torch.Tensor, antonyms))
    synonyms: torch.Tensor = shuffle(cast(torch.Tensor, synonyms))
    synonyms = synonyms[:na]  # chop synonyms to length of antonyms
    inputs: torch.Tensor = torch.stack([antonyms, synonyms], dim=0)

    train_words = inputs[:, : args.n_train]  # equal count of antonyms and synonyms

    # now exclude all words from test that appear in train
    train_words = train_words.reshape(4 * args.n_train, d)
    # inputs = inputs.reshape(-1, d)
    inputs = inputs.reshape(-1, 2, d)
    equal = cast(
        torch.Tensor,
        inputs.unsqueeze(-2) == train_words.reshape(1, 1, len(train_words), d),
    )
    equal: torch.Tensor = equal.all(-1)
    is_train = equal.any(-1)  # equals any train_word
    is_train = is_train.any(-1)  # either lemma or counterpart equals some train_word
    is_test = ~is_train

    targets = torch.cat([torch.ones(na), torch.zeros(na)])

    # split according to is_train, is_test
    train_dataset = create_dataset(inputs, targets, is_train)
    test_dataset = create_dataset(inputs, targets, is_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    embedding_size = GPT2Config.from_pretrained(args.model_name).n_embd

    train_inputs = cast(torch.Tensor, train_dataset.inputs)
    train_inputs = train_inputs.to(device)
    d = train_inputs.size(-1)

    encoder = nn.Sequential(
        Lambda(lambda x: x.reshape(-1, d)),
        nn.EmbeddingBag(int(inputs.max()), embedding_size)
        if args.architecture == BASELINE
        else GPTEmbed(
            model_name=args.model_name,
            randomize_parameters=args.architecture == UNTRAINED,
            train_ln=args.train_ln,
            train_wpe=args.train_wpe,
        ),
        Lambda(lambda x: x.reshape(-1, 2, embedding_size)),
    ).to(device)

    with torch.no_grad():
        outputs = encoder(train_inputs)

    mean = outputs.mean(dim=0, keepdims=True)
    std = outputs.mean(dim=0, keepdims=True)

    if args.architecture in [BINARY_PRETRAINED, BINARY_UNTRAINED]:
        encoder = nn.Sequential(
            encoder,
            Lambda(lambda x: (x - mean) / std),
            nn.Sigmoid(),
            *([] if args.train_ln or args.train_wpe else [Lambda(lambda x: x.round())]),
        )

    model = Net(
        encoder=encoder,
        embedding_size=embedding_size,
        hidden_size=args.hidden_size,
        multiplicative_interaction=args.multiplicative_interaction,
    ).to(device)

    start_time = time.time()
    frames = 0
    save_count = 0

    def evaluate(_epoch):
        test_loss = 0
        _correct = []
        with torch.no_grad():
            for _data, _target in test_loader:
                _data, _target = _data.to(device), _target.to(device)
                _output = model(_data)
                test_loss += F.binary_cross_entropy(
                    _output, _target, reduction="sum"
                ).item()  # sum up batch loss
                _pred = _output.round()  # get the index of the max log-probability
                _correct += [_pred.eq(_target.view_as(_pred)).float()]
        test_loss /= len(test_loader.dataset)
        test_accuracy = torch.cat(_correct).mean()
        _log = {
            EPOCH: _epoch,
            HOURS: (time.time() - start_time) / 3600,
            TEST_LOSS: test_loss,
            TEST_ACCURACY: test_accuracy.item(),
            RUN_ID: logger.run_id,
        }
        pprint(_log)
        if logger.run_id is not None:
            logger.log(_log)

    save_path = get_save_path(logger.run_id)
    if args.load_id is not None:
        load_path = get_save_path(args.load_id)
        logging.info(f"Loading checkpoint from {load_path}...")
        model.load_state_dict(torch.load(load_path))
    if args.save_interval is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(args.epochs):
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), str(save_path))
            save_count += 1

        correct = []
        for batch_idx, (data, target) in enumerate(train_loader):
            frames += len(data)
            if batch_idx == 0:
                evaluate(epoch)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            pred = output.round()
            correct += [pred.eq(target.view_as(pred)).float()]
            if batch_idx == 0 and batch_idx % args.log_interval == 0:
                accuracy = torch.cat(correct).mean()
                seconds = time.time() - start_time
                log = {
                    EPOCH: epoch,
                    HOURS: seconds / 3600,
                    FPS: frames / seconds,
                    LOSS: loss.item(),
                    ACCURACY: accuracy.item(),
                    RUN_ID: logger.run_id,
                    SAVE_COUNT: save_count,
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
TEST_ACCURACY = "test accuracy"
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
