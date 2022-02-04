import codecs
import functools
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import List, Literal, Optional, cast

import gym
import numpy as np
import sweep_logger
import torch
import utils
from agent import Agent
from envs import TimeLimitMask, TransposeImage, VecPyTorch, VecPyTorchFrameStack
from gym.wrappers.clip_action import ClipAction
from line_chart import spec
from ppo import PPO
from rollouts import Rollouts
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sweep_logger import HasuraLogger
from tap import Tap

try:
    # noinspection PyUnresolvedReferences
    import dmc2gym
except ImportError:
    pass


class InvalidEnvId(RuntimeError):
    pass


ACTION_LOSS = "action loss"
ENTROPY = "entropy"
EPISODE_RETURN = "episode return"
EPISODE_LENGTH = "episode length"
FPS = "fps"
GRADIENT_NORM = "gradient norm"
HOURS = "hours"
SAVE_COUNT = "save count"
STEP = "step"
TEST_EPISODE_RETURN = "test episode return"
TEST_EPISODE_LENGTH = "test episode length"
VALUE_LOSS = "value loss"
TIME = "time"


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
    allow_early_resets: bool = False
    alpha: float = 0.99  # Adam alpha
    clip_param: float = 0.1  # PPO clip parameter
    config: Optional[str] = None  # If given, yaml config from which to load params
    cuda: bool = True  # enable CUDA
    entropy_coef: float = 0.01  # auxiliary entropy objective coefficient
    env: str = "BreakoutNoFrameskip-v4"  # env ID for gym
    eps: float = 1e-5  # RMSProp epsilon
    gae: bool = True  # use Generalized Advantage Estimation
    gae_lambda: float = 0.95  # GAE lambda parameter
    gamma: float = 0.99  # discount factor
    graphql_endpoint: str = os.getenv("GRAPHQL_ENDPOINT")
    hidden_size: int = 256
    host_machine: str = os.getenv("HOST_MACHINE")
    log_interval: int = 100  # how many updates to log between
    linear_lr_decay: bool = False  # anneal the learning rate
    load_id: int = None  # path to load parameters from if at all
    log_level: str = "INFO"
    lr: float = 2.5e-4  # learning rate
    max_grad_norm: float = 0.5  # clip gradient norms
    num_env_steps: int = 1e9  # total number of environment steps
    num_mini_batch: int = 4  # number of mini-batches per update
    num_processes: int = 8  # number of parallel environments
    num_steps: int = 128  # number of forward steps in A2C
    ppo_epoch: int = 3  # number of PPO updates
    recurrent: bool = False  # use recurrence in the policy
    render: bool = False
    render_test: bool = False
    save_interval: Optional[int] = None  # how many updates to save between
    seed: int = 0  # random seed
    sync_envs: bool = False
    test_interval: Optional[int] = None  # how many updates to evaluate between
    use_proper_time_limits: bool = False  # compute returns with time limits
    value_coef: float = 1  # value loss coefficient
    visualizer_url: str = os.getenv("VISUALIZER_URL")

    def configure(self) -> None:
        self.add_subparsers(dest="logger_args")
        configure_logger_args(self)


class ArgsType(Args):
    logger_args: Optional[RUN_OR_SWEEP]


@dataclass
class Counters:
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_successes: List[int] = field(default_factory=list)

    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []


@dataclass
class TimeSteps:
    action: np.ndarray
    observation: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: List[dict]


class Trainer:
    @classmethod
    def build_counters(cls):
        return Counters()

    @classmethod
    def charts(cls, **kwargs):
        return [
            *[
                spec(x=HOURS, y=y, **kwargs)
                for y in (EPISODE_RETURN, TEST_EPISODE_RETURN)
            ],
            *[
                spec(x=STEP, y=y, **kwargs)
                for y in (
                    EPISODE_RETURN,
                    TEST_EPISODE_RETURN,
                    FPS,
                    ENTROPY,
                    GRADIENT_NORM,
                    SAVE_COUNT,
                )
            ],
        ]

    @classmethod
    def cuda(cls, args):
        return args.cuda and torch.cuda.is_available()

    @classmethod
    def device(cls, cuda):
        return torch.device(torch.cuda.device_count() - 1 if cuda else "cpu")

    @classmethod
    def evaluate(
        cls, agent, envs, num_processes, device, start, total_num_steps, logger, test
    ):

        episode_rewards = []
        episode_lengths = []
        episode_success = []

        obs = envs.reset()
        recurrent_hidden_states = torch.zeros(
            num_processes, agent.recurrent_hidden_state_size, device=device
        )
        masks = torch.zeros(num_processes, 1, device=device)

        while len(episode_rewards) < 100:
            with torch.no_grad():
                _, action, _, recurrent_hidden_states = agent.forward(
                    obs, recurrent_hidden_states, masks
                )

            # Observe reward and next obs
            obs, rewards, done, infos = envs.step(action)
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device,
            )

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                if "success" in info.keys():
                    episode_success.append(info["success"])

        envs.close()
        now = time.time()
        log = {
            TIME: now * 1000000,
            HOURS: (now - start) / 3600,
            STEP: total_num_steps,
        }
        if test:
            log.update(
                {
                    TEST_EPISODE_RETURN: np.mean(episode_rewards),
                    TEST_EPISODE_LENGTH: np.mean(episode_lengths),
                }
            )

        logging.info(pformat(log))
        if logger.run_id is not None:
            log.update({"run ID": logger.run_id})
        logging.info(pformat(log))
        if logger.run_id is not None:
            logger.log(log)

        logging.info(
            " Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(episode_rewards), np.mean(episode_rewards)
            )
        )

    @classmethod
    def excluded(cls):
        return {
            "config",
            "host_machine",
            "load_id",
            "logger_args",
            "name",
            "render",
            "render_test",
            "subcommand",
            "sweep_id",
            "sync_envs",
        }

    @staticmethod
    def load(agent, load_path):
        agent.load_state_dict(torch.load(load_path))

    @classmethod
    def log(
        cls,
        logger: HasuraLogger,
        log: dict,
        counters: Counters,
        total_num_steps: int,
    ):
        logging.info(pformat(log))
        if logger.run_id is not None:
            logger.log(log)

    @staticmethod
    def num_eval_processes(args):
        return args.num_processes

    @classmethod
    def process_info(cls, counters: Counters, info: dict):
        if "episode" in info.keys():
            counters.episode_rewards.append(info["episode"]["r"])
            counters.episode_lengths.append(info["episode"]["l"])

    @staticmethod
    def total_num_steps(j, args):
        return j * args.num_processes * args.num_steps

    @staticmethod
    def blob(logger: HasuraLogger, blob, metadata: dict):
        tick = time.time()

        # https://stackoverflow.com/a/30469744/4176597
        pickled = codecs.encode(pickle.dumps(blob), "base64").decode()

        logger.blob(blob=pickled, metadata=metadata)
        logging.info(f"Sending blob took {time.time() - tick} seconds.")

    @staticmethod
    def make_env(env, seed, allow_early_resets, render: bool = False, **kwargs):
        def _thunk(env_id):
            if env_id.startswith("dm"):
                _, domain, task = env_id.split(".")
                env = dmc2gym.make(domain_name=domain, task_name=task)
                env = ClipAction(env)
            else:
                env = gym.make(env_id)

            is_atari = hasattr(gym.envs, "atari") and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv
            )
            if is_atari:
                env = NoopResetEnv(env, noop_max=30)
                env = MaxAndSkipEnv(env, skip=4)

            env.seed(seed)

            if str(env.__class__.__name__).find("TimeLimit") >= 0:
                env = TimeLimitMask(env)

            env = Monitor(env, allow_early_resets=allow_early_resets)

            if is_atari:
                if len(env.observation_space.shape) == 3:
                    env = EpisodicLifeEnv(env)
                    if "FIRE" in env.unwrapped.get_action_meanings():
                        env = FireResetEnv(env)
                    env = WarpFrame(env, width=84, height=84)
                    env = ClipRewardEnv(env)
            elif len(env.observation_space.shape) == 3:
                raise NotImplementedError(
                    "CNN models work only for atari,\n"
                    "please use a custom wrapper for a custom pixel input env.\n"
                    "See wrap_deepmind for an example."
                )

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

            return env

        return functools.partial(_thunk, env_id=env)

    @staticmethod
    def make_agent(envs: VecPyTorch, args) -> Agent:
        obs_shape = envs.observation_space.shape
        action_space = envs.action_space
        return Agent(
            obs_shape=obs_shape,
            action_space=action_space,
            recurrent=args.recurrent,
        )

    @classmethod
    def make_vec_envs(
        cls,
        device: torch.device,
        num_processes: int,
        render: bool,
        render_test: bool,
        seed: int,
        sync_envs: bool,
        test: bool,
        num_frame_stack: int = None,
        **kwargs,
    ):
        if test:
            render = render_test

        envs = [
            cls.make_env(rank=i, render=render, seed=seed + i, test=test, **kwargs)
            for i in range(num_processes)
        ]

        if len(envs) > 1 and not sync_envs:
            envs = SubprocVecEnv(envs)
        else:
            envs = DummyVecEnv(envs)

        envs = VecPyTorch(envs, device)

        if num_frame_stack is not None:
            envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        elif len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

        return envs

    @staticmethod
    def save(agent, save_path: Path, args: Args):
        torch.save(agent.state_dict(), save_path)

    @staticmethod
    def save_dir(run_id: Optional[int] = None):
        path = Path("/tmp/logs")
        if run_id is not None:
            return Path(path, str(run_id))
        return path

    @classmethod
    def save_path(cls, run_id: Optional[int] = None):
        return Path(cls.save_dir(run_id), "checkpoint.pkl")

    @classmethod
    def update_args(cls, args, parameters, check_hasattr=True):
        for k, v in parameters.items():
            if k not in cls.excluded():
                if check_hasattr:
                    assert hasattr(args, k), k
                setattr(args, k, v)

    @classmethod
    def train(cls, args: Args, logger: HasuraLogger):
        logging.info(pformat(args.as_dict()))

        render = args.render or args.render_test
        if render:
            args.num_processes = 1

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

        cuda = cls.cuda(args)
        if cuda:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        torch.set_num_threads(1)
        device = cls.device(cuda)

        envs = cls.make_vec_envs(
            device=device, run_id=logger.run_id, test=False, **args.as_dict()
        )
        try:

            agent = cls.make_agent(envs=envs, args=args)
            if args.load_id is not None:
                load_path = cls.save_path(args.load_id)
                logging.info(f"Loading checkpoint from {load_path}...")
                cls.load(agent, load_path)
            agent.to(device)

            ppo = PPO(
                agent=agent,
                clip_param=args.clip_param,
                ppo_epoch=args.ppo_epoch,
                num_mini_batch=args.num_mini_batch,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
            )

            rollouts = Rollouts(
                num_steps=args.num_steps,
                num_processes=args.num_processes,
                obs_shape=envs.observation_space.shape,
                action_space=envs.action_space,
                recurrent_hidden_state_size=agent.recurrent_hidden_state_size,
            )

            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)

            counters = cls.build_counters()

            tick = start = time.time()
            num_steps = 0
            save_count = 0
            num_updates = (
                int(args.num_env_steps) // args.num_steps // args.num_processes
            )
            for j in range(num_updates):
                if args.test_interval is not None and j % args.test_interval == 0:
                    cls.evaluate(
                        agent=agent,
                        envs=cls.make_vec_envs(
                            device=device,
                            run_id=logger.run_id,
                            test=True,
                            **args.as_dict(),
                        ),
                        num_processes=cls.num_eval_processes(args),
                        device=device,
                        start=start,
                        total_num_steps=cls.total_num_steps(j, args),
                        logger=logger,
                        test=True,
                    )
                # save for every interval-th episode or for the last epoch
                if (
                    args.save_interval is not None
                    and logger.run_id is not None
                    and (j % args.save_interval == 0 or j == num_updates - 1)
                    and not render
                ):
                    save_path = cls.save_path(logger.run_id)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cls.save(agent, save_path, args)
                    save_count += 1

                if args.linear_lr_decay:
                    # decrease learning rate linearly
                    utils.update_linear_schedule(ppo.optimizer, j, num_updates, args.lr)

                for step in range(args.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        (
                            value,
                            action,
                            action_log_prob,
                            recurrent_hidden_states,
                        ) = agent.forward(
                            inputs=rollouts.obs[step],
                            rnn_hxs=rollouts.recurrent_hidden_states[step],
                            masks=rollouts.masks[step],
                        )

                    # Observe reward and next obs
                    obs, reward, done, infos = envs.step(action)

                    for info in infos:
                        cls.process_info(counters, info)

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done]
                    )
                    bad_masks = torch.FloatTensor(
                        [
                            [0.0] if "bad_transition" in info.keys() else [1.0]
                            for info in infos
                        ]
                    )
                    rollouts.insert(
                        obs=obs,
                        recurrent_hidden_states=recurrent_hidden_states,
                        actions=action,
                        action_log_probs=action_log_prob,
                        value_preds=value,
                        rewards=reward,
                        masks=masks,
                        bad_masks=bad_masks,
                    )

                with torch.no_grad():
                    next_value = agent.get_value(
                        inputs=rollouts.obs[-1],
                        rnn_hxs=rollouts.recurrent_hidden_states[-1],
                        masks=rollouts.masks[-1],
                    ).detach()

                rollouts.compute_returns(
                    next_value=next_value,
                    use_gae=args.gae,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    use_proper_time_limits=args.use_proper_time_limits,
                )

                total_num_steps = cls.total_num_steps(j + 1, args)
                if not render:
                    value_loss, action_loss, dist_entropy, gradient_norm = ppo.update(
                        rollouts
                    )

                    rollouts.after_update()

                    if j % args.log_interval == 0:
                        now = time.time()
                        fps = (total_num_steps - num_steps) / (now - tick)
                        tick = time.time()
                        num_steps = total_num_steps
                        log = {
                            ACTION_LOSS: action_loss,
                            ENTROPY: dist_entropy,
                            EPISODE_LENGTH: np.mean(counters.episode_lengths),
                            EPISODE_RETURN: np.mean(counters.episode_rewards),
                            FPS: fps,
                            GRADIENT_NORM: gradient_norm,
                            HOURS: (now - start) / 3600,
                            SAVE_COUNT: save_count,
                            STEP: total_num_steps,
                            TIME: now * 1000000,
                            VALUE_LOSS: value_loss,
                        }

                        logging.info(pformat(log))
                        if logger.run_id is not None:
                            log.update({"run ID": logger.run_id})

                        cls.log(logger, log, counters, total_num_steps)
                        counters.reset()
        finally:
            envs.close()

    @classmethod
    def main(cls, args: ArgsType):
        logging.getLogger().setLevel(args.log_level)
        kwargs = dict(visualizer_url=args.visualizer_url)

        charts = cls.charts(**kwargs)

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
        cls.update_args(args, params)
        return cls.train(args=args, logger=logger)


if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
