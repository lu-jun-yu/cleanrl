# OPTS_TTPO (On-Policy Parallel Tree Search + Tree Trajectory Policy Optimization) for Atari
# Based on PPO implementation from CleanRL
import os
import random
import time
import json
from collections import defaultdict
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from opts_ttpo_core_exp5_2 import compute_branch_weight, compute_tree_gae, select_next_states


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    resume: bool = False
    """whether to resume training from a checkpoint"""
    resume_path: str = ""
    """checkpoint path to resume from; defaults to checkpoint_dir/env_id/seed{seed}.pt"""
    save_checkpoint: bool = True
    """whether to save a resumable training checkpoint"""
    checkpoint_dir: str = "checkpoints/atari"
    """root directory for resumable training checkpoints"""
    checkpoint_interval: int = 100
    """save a checkpoint every N iterations"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    tau: float = 0.7
    """tau for the OTRC node selection"""
    max_search_per_tree: int = 4
    """maximum number of tree searches per environment per iteration"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class AtariStateSnapshotWrapper(gym.Wrapper):
    """Wrapper to support state snapshots for Atari environments using ALE.

    Saves and restores:
    - ALE simulator state (RAM, registers, etc.)
    - FrameStack buffer (critical for correct observations)
    - MaxAndSkipEnv observation buffer
    - EpisodicLifeEnv life tracking state
    - TimeLimit elapsed steps
    - RecordEpisodeStatistics counters
    """

    def __init__(self, env):
        super().__init__(env)
        self.ale = self.unwrapped.ale

        # Find all wrappers that need state saved/restored
        self._timelimit_wrapper = None
        self._record_stats_wrapper = None
        self._framestack_wrapper = None
        self._maxandskip_wrapper = None
        self._episodiclife_wrapper = None

        current = env
        while current is not None:
            if hasattr(current, '_elapsed_steps'):  # TimeLimit wrapper
                self._timelimit_wrapper = current
            if hasattr(current, 'episode_returns'):  # RecordEpisodeStatistics wrapper
                self._record_stats_wrapper = current
            if hasattr(current, 'frames') and hasattr(current, 'num_stack'):  # FrameStack wrapper
                self._framestack_wrapper = current
            if hasattr(current, '_obs_buffer') and hasattr(current, '_skip'):  # MaxAndSkipEnv wrapper
                self._maxandskip_wrapper = current
            if hasattr(current, 'lives') and hasattr(current, 'was_real_done'):  # EpisodicLifeEnv wrapper
                self._episodiclife_wrapper = current
            current = getattr(current, 'env', None)

        # Track RecordEpisodeStatistics values for state snapshot
        # We need to capture these BEFORE RecordEpisodeStatistics resets them on terminal
        self._episode_return_snapshot = 0.0
        self._episode_length_snapshot = 0

    def step(self, action):
        """Step the environment and capture episode statistics for snapshot."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Keep a best-effort scalar mirror for diagnostics. Clone/restore uses the
        # wrapper's full arrays directly because EpisodicLifeEnv can emit terminal
        # signals on life loss while RecordEpisodeStatistics must keep counting.
        if self._record_stats_wrapper is not None and self._record_stats_wrapper.episode_returns is not None:
            self._episode_return_snapshot = float(self._record_stats_wrapper.episode_returns[0])
            self._episode_length_snapshot = int(self._record_stats_wrapper.episode_lengths[0])

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and snapshot tracking."""
        obs, info = self.env.reset(**kwargs)
        # Reset snapshot
        self._episode_return_snapshot = 0.0
        self._episode_length_snapshot = 0
        return obs, info

    def clone_state(self):
        """Clone the current environment state including all wrapper states."""
        if hasattr(self.ale, "cloneSystemState"):
            ale_state = ("system", self.ale.cloneSystemState())
        else:
            ale_state = ("emulator", self.ale.cloneState())

        # Save TimeLimit state
        timelimit_steps = None
        if self._timelimit_wrapper is not None:
            timelimit_steps = self._timelimit_wrapper._elapsed_steps

        # Save RecordEpisodeStatistics state. Store the real wrapper arrays instead
        # of the scalar mirror because life-loss terminals do not reset real episodes.
        record_stats = None
        if self._record_stats_wrapper is not None and self._record_stats_wrapper.episode_returns is not None:
            record_stats = {
                'episode_returns': np.array(self._record_stats_wrapper.episode_returns, copy=True),
                'episode_lengths': np.array(self._record_stats_wrapper.episode_lengths, copy=True),
                'episode_start_times': (
                    np.array(self._record_stats_wrapper.episode_start_times, copy=True)
                    if self._record_stats_wrapper.episode_start_times is not None
                    else None
                ),
                'episode_count': int(self._record_stats_wrapper.episode_count),
                'return_queue': list(self._record_stats_wrapper.return_queue),
                'length_queue': list(self._record_stats_wrapper.length_queue),
            }

        # Save FrameStack buffer (critical for correct observations)
        framestack_state = None
        if self._framestack_wrapper is not None:
            # self.frames is a deque of LazyFrames or arrays
            frames_list = []
            for frame in self._framestack_wrapper.frames:
                if hasattr(frame, '__array__'):
                    frames_list.append(np.array(frame, copy=True))
                else:
                    frames_list.append(frame)
            framestack_state = frames_list

        # Save MaxAndSkipEnv buffer
        maxandskip_state = None
        if self._maxandskip_wrapper is not None:
            maxandskip_state = [
                np.array(obs, copy=True) if obs is not None else None
                for obs in self._maxandskip_wrapper._obs_buffer
            ]

        # Save EpisodicLifeEnv state
        episodiclife_state = None
        if self._episodiclife_wrapper is not None:
            episodiclife_state = {
                'lives': self._episodiclife_wrapper.lives,
                'was_real_done': self._episodiclife_wrapper.was_real_done,
            }

        # Save underlying RNG state (NoopResetEnv samples no-op counts from it)
        rng_state = None
        if self.unwrapped.np_random is not None:
            rng_state = self.unwrapped.np_random.bit_generator.state

        return (ale_state, timelimit_steps, record_stats, framestack_state, maxandskip_state, episodiclife_state, rng_state)

    def restore_state(self, state):
        """Restore the environment to a previous state including all wrapper states."""
        ale_state, timelimit_steps, record_stats, framestack_state, maxandskip_state, episodiclife_state = state[:6]
        rng_state = state[6] if len(state) > 6 else None

        # Restore ALE state first
        ale_state_type, ale_state_payload = ale_state
        if ale_state_type == "system" and hasattr(self.ale, "restoreSystemState"):
            self.ale.restoreSystemState(ale_state_payload)
        else:
            self.ale.restoreState(ale_state_payload)

        # Restore TimeLimit state
        if self._timelimit_wrapper is not None and timelimit_steps is not None:
            self._timelimit_wrapper._elapsed_steps = timelimit_steps

        # Restore RecordEpisodeStatistics state using array indexing
        if self._record_stats_wrapper is not None and record_stats is not None:
            self._record_stats_wrapper.episode_returns = np.array(record_stats['episode_returns'], copy=True)
            self._record_stats_wrapper.episode_lengths = np.array(record_stats['episode_lengths'], copy=True)
            self._record_stats_wrapper.episode_start_times = (
                np.array(record_stats['episode_start_times'], copy=True)
                if record_stats['episode_start_times'] is not None
                else None
            )
            self._record_stats_wrapper.episode_count = int(record_stats['episode_count'])
            self._record_stats_wrapper.return_queue.clear()
            self._record_stats_wrapper.return_queue.extend(record_stats['return_queue'])
            self._record_stats_wrapper.length_queue.clear()
            self._record_stats_wrapper.length_queue.extend(record_stats['length_queue'])
            # Also update our snapshot
            self._episode_return_snapshot = float(self._record_stats_wrapper.episode_returns[0])
            self._episode_length_snapshot = int(self._record_stats_wrapper.episode_lengths[0])

        # Restore FrameStack buffer (critical for correct observations)
        if self._framestack_wrapper is not None and framestack_state is not None:
            self._framestack_wrapper.frames.clear()
            for frame in framestack_state:
                self._framestack_wrapper.frames.append(frame)

        # Restore MaxAndSkipEnv buffer
        if self._maxandskip_wrapper is not None and maxandskip_state is not None:
            for i, obs in enumerate(maxandskip_state):
                if obs is not None:
                    self._maxandskip_wrapper._obs_buffer[i] = np.array(obs, copy=True)
                else:
                    self._maxandskip_wrapper._obs_buffer[i] = None

        # Restore EpisodicLifeEnv state
        if self._episodiclife_wrapper is not None and episodiclife_state is not None:
            self._episodiclife_wrapper.lives = episodiclife_state['lives']
            self._episodiclife_wrapper.was_real_done = episodiclife_state['was_real_done']

        # Restore underlying RNG state
        if rng_state is not None and self.unwrapped.np_random is not None:
            self.unwrapped.np_random.bit_generator.state = rng_state


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        make_kwargs = {}
        if env_id.startswith("ALE/") or env_id.endswith("-v5"):
            # CleanRL Atari preprocessing expects NoFrameskip-like deterministic
            # base envs; MaxAndSkipEnv below provides the training frameskip.
            make_kwargs = {"frameskip": 1, "repeat_action_probability": 0.0}
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **make_kwargs)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **make_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = AtariStateSnapshotWrapper(env)
        return env

    return thunk


def get_checkpoint_path(args):
    return os.path.join(args.checkpoint_dir, args.env_id, f"seed{args.seed}.pt")


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(path, args, run_name, iteration, global_step, agent, optimizer, envs, next_obs, next_done):
    checkpoint = {
        "version": 1,
        "args": vars(args),
        "run_name": run_name,
        "iteration": int(iteration),
        "global_step": int(global_step),
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "env_states": [env.clone_state() for env in envs],
        "next_obs": next_obs.detach().cpu(),
        "next_done": next_done.detach().cpu(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"checkpoint saved to {path}")


def load_training_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    checkpoint_path = args.resume_path if args.resume_path else get_checkpoint_path(args)
    resume_checkpoint = None
    if args.resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        resume_checkpoint = load_training_checkpoint(checkpoint_path, torch.device("cpu"))
        print(
            f"resuming from {checkpoint_path} "
            f"(iteration={resume_checkpoint['iteration']}, global_step={resume_checkpoint['global_step']})"
        )

    run_name = (
        resume_checkpoint.get("run_name")
        if resume_checkpoint is not None and resume_checkpoint.get("run_name")
        else f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
    algorithm_name = f"{args.exp_name}_tau{args.tau}_s{args.max_search_per_tree}_20260807"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup - use independent environment list for state snapshot support
    envs = [make_env(args.env_id, i, args.capture_video, run_name)() for i in range(args.num_envs)]
    assert isinstance(envs[0].action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Agent only needs CleanRL-style single_*_space attributes for network shapes.
    # Use the training env spaces directly to avoid constructing a second set of envs.
    agent_env_spaces = type(
        "AgentEnvSpaces",
        (),
        {
            "single_observation_space": envs[0].observation_space,
            "single_action_space": envs[0].action_space,
        },
    )()
    agent = Agent(agent_env_spaces).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs[0].observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs[0].action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # OPTS_TTPO: Tree structure tensors
    parent_indices = -torch.ones((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    # OPTS_TTPO: Tree id per node (used when nodes from the same tree are not contiguous)
    tree_indices = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    state_branches = torch.ones((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    advantages = torch.zeros((args.num_steps, args.num_envs)).to(device)
    returns = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # OPTS_TTPO: Non-tensor data
    env_states = [[None for _ in range(args.num_envs)] for _ in range(args.num_steps)]

    # OPTS_TTPO: Root node data
    root_states = [[] for _ in range(args.num_envs)]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.zeros((args.num_envs,) + envs[0].observation_space.shape).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    start_iteration = 1

    if resume_checkpoint is None:
        for env_idx, env in enumerate(envs):
            obs_data, _ = env.reset(seed=args.seed + env_idx)
            next_obs[env_idx] = torch.Tensor(obs_data).to(device)
            root_states[env_idx] = [env.clone_state()]
    else:
        required_match_keys = [
            "env_id", "seed", "num_envs", "num_steps", "num_minibatches",
            "gamma", "gae_lambda", "tau", "max_search_per_tree",
        ]
        mismatched = [k for k in required_match_keys if resume_checkpoint["args"].get(k) != getattr(args, k)]
        if mismatched:
            details = ", ".join(
                f"{k}: checkpoint={resume_checkpoint['args'].get(k)} current={getattr(args, k)}" for k in mismatched
            )
            raise ValueError(f"resume checkpoint args must match current run; mismatched -> {details}")
        agent.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)

        for env, env_state in zip(envs, resume_checkpoint["env_states"]):
            env.reset()
            env.restore_state(env_state)

        next_obs = resume_checkpoint["next_obs"].to(device)
        next_done = resume_checkpoint["next_done"].to(device)
        global_step = int(resume_checkpoint["global_step"])
        start_iteration = int(resume_checkpoint["iteration"]) + 1

        random.setstate(resume_checkpoint["python_random_state"])
        np.random.set_state(resume_checkpoint["numpy_random_state"])
        torch.set_rng_state(resume_checkpoint["torch_random_state"].cpu())
        cuda_rng_state = resume_checkpoint.get("torch_cuda_random_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

        for env_idx, env in enumerate(envs):
            root_states[env_idx] = [env.clone_state()]

    session_start_step = global_step

    # weighted loss aggregation; constant normalizer across minibatches
    def wagg(t):
        return (t * w).sum() / loss_norm

    for iteration in range(start_iteration, args.num_iterations + 1):
        episodic_return_info = []  # (episodic_return, tid, step, env_idx)

        # OPTS_TTPO: root_branch_counts maintained incrementally
        root_branch_counts = [defaultdict(int) for _ in range(args.num_envs)]

        # search count per tree (reset each iteration)
        search_count = [{} for _ in range(args.num_envs)]

        # pooled mean-otrc_score stats for tree filtering (verify_scaling_variance v2)
        max_otrc_scores = [{} for _ in range(args.num_envs)]
        tree_search_state = [{} for _ in range(args.num_envs)]

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if iteration == 1:
            skip_init_search = [False] * args.num_envs
        else:
            skip_init_search = [True] * args.num_envs
        for env_idx, env in enumerate(envs):
            if next_done[env_idx] == 1:
                skip_init_search[env_idx] = False
                obs_data, _ = env.reset()
                next_obs[env_idx] = torch.Tensor(obs_data).to(device)
            root_states[env_idx] = [env.clone_state()]
        next_done.zero_()

        current_parent = [-1] * args.num_envs
        state_branches.fill_(1)
        advantages.zero_()
        parent_indices.fill_(-1)
        tree_indices.zero_()

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # Save current obs
            obs[step] = next_obs

            # Sample actions
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Execute actions in all environments
            next_obs_list = []
            rewards_list = []
            next_done_list = []

            # Save parent_indices and update root_branch_counts
            for env_idx in range(args.num_envs):
                p = current_parent[env_idx]
                parent_indices[step, env_idx] = p
                tree_indices[step, env_idx] = p if p < 0 else tree_indices[p, env_idx]

                # Update root_branch_counts if this is a root node
                if p < 0:
                    root_branch_counts[env_idx][p] += 1

                current_parent[env_idx] = step

            for env_idx, env in enumerate(envs):
                next_o, r, terminated, truncated, info = env.step(action[env_idx].cpu().numpy())
                next_obs_list.append(next_o)
                rewards_list.append(r)
                next_done_list.append(terminated or truncated)

                # Save state snapshot (after action)
                env_states[step][env_idx] = env.clone_state()

                # Log episode info
                if (terminated or truncated) and "episode" in info:
                    ep_r = info['episode']['r']
                    ep_l = info['episode']['l']
                    episodic_return = float(ep_r[0] if hasattr(ep_r, '__getitem__') else ep_r)
                    episodic_length = int(ep_l[0] if hasattr(ep_l, '__getitem__') else ep_l)
                    episodic_return_info.append((episodic_return, tree_indices[step, env_idx].item(), step, env_idx))
                    print(f"global_step={global_step}, episodic_return={episodic_return:.4f}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

            # Convert to tensors
            next_obs = torch.stack([torch.Tensor(o) for o in next_obs_list]).to(device)
            rewards[step] = torch.tensor(rewards_list).to(device)
            next_done = torch.tensor(next_done_list, dtype=torch.float32).to(device)

            # Update dones[step] to reflect terminal status after this step's action
            dones[step] = next_done

            # Handle terminated environments
            terminated_envs = [i for i in range(args.num_envs) if next_done_list[i]]

            if len(terminated_envs) > 0:
                # TreeGAE update
                for env_idx in terminated_envs:
                    compute_tree_gae(
                        terminal_step=step,
                        env_idx=env_idx,
                        rewards=rewards,
                        values=values,
                        dones=dones,
                        parent_indices=parent_indices,
                        advantages=advantages,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                    )

                if step < args.num_steps - 1:
                    affected_tree_ids = [int(tree_indices[step, env_idx].item()) for env_idx in terminated_envs]
                    selected = select_next_states(
                        terminated_envs=terminated_envs,
                        current_step=step,
                        advantages=advantages,
                        parent_indices=parent_indices,
                        tree_indices=tree_indices,
                        search_count=search_count,
                        max_search=args.max_search_per_tree,
                        max_otrc_scores=max_otrc_scores,
                        skip_init_search=skip_init_search,
                        tree_search_state=tree_search_state,
                        affected_tree_ids=affected_tree_ids,
                        gamma=args.gamma,
                        tau=args.tau,
                    )

                    # OTRC selection and state restoration
                    for i, env_idx in enumerate(terminated_envs):
                        if selected[i] < 0:
                            # Variance is stable, start a new tree
                            obs_data, _ = envs[env_idx].reset()
                            next_obs[env_idx] = torch.Tensor(obs_data).to(device)
                            root_states[env_idx].insert(0, envs[env_idx].clone_state())
                            current_parent[env_idx] = -len(root_states[env_idx])
                        else:
                            # Continue searching
                            parent = parent_indices[selected[i], env_idx].item()
                            if parent < 0:
                                envs[env_idx].restore_state(root_states[env_idx][parent])
                            else:
                                envs[env_idx].restore_state(env_states[parent][env_idx])
                                state_branches[parent, env_idx] += 1
                            next_obs[env_idx] = obs[selected[i], env_idx]
                            current_parent[env_idx] = parent

        # Bootstrap value for non-terminal leaf nodes and recompute advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()

        # Final TreeGAE update for non-terminated trajectories
        for env_idx in range(args.num_envs):
            if current_parent[env_idx] >= 0:
                compute_tree_gae(
                    terminal_step=current_parent[env_idx],
                    env_idx=env_idx,
                    rewards=rewards,
                    values=values,
                    dones=dones,
                    parent_indices=parent_indices,
                    advantages=advantages,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    next_value=next_value[env_idx].item(),
                )

        # Compute returns: returns[t] = A(s_t, a_t) + V(s_t)
        returns = advantages + values

        # Compute direct branch weights for all environments
        branch_weights = compute_branch_weight(
            num_steps=args.num_steps,
            parent_indices=parent_indices,
            env_indices=list(range(args.num_envs)),
        )

        # Compute tree-weighted aggregated returns
        if episodic_return_info:
            tid_groups = defaultdict(list)  # (env_idx, tid) -> [(return, step)]
            for ep_return, tid, ep_step, ei in episodic_return_info:
                tid_groups[(ei, tid)].append((ep_return, ep_step))

            aggregated_returns = []
            for (ei, tid), entries in tid_groups.items():
                weighted_sum = 0.0
                weight_sum = 0.0
                for ep_return, ep_step in entries:
                    w = branch_weights[ep_step, ei].item()
                    weighted_sum += ep_return * w
                    weight_sum += w
                aggregated_returns.append(weighted_sum / weight_sum if weight_sum > 0 else 0.0)

            mean_return = sum(aggregated_returns) / len(aggregated_returns)
            max_return = max(aggregated_returns)
            min_return = min(aggregated_returns)
        else:
            mean_return = 0.0
            max_return = 0.0
            min_return = 0.0

        print(f"Iteration {iteration}: mean_return={mean_return:.4f}, max_return={max_return:.4f}, min_return={min_return:.4f}")

        # Save results to JSON file
        folder_name = f"./results/{args.num_envs}_{args.num_steps}/{algorithm_name}"
        os.makedirs(folder_name, exist_ok=True)
        safe_env_id = args.env_id.replace("/", "_")
        result_filename = f"{folder_name}/{safe_env_id}_{args.seed}.json"
        if os.path.exists(result_filename):
            with open(result_filename, "r") as f:
                results_data = json.load(f)
        else:
            results_data = []
        results_data.append({
            "step": str(global_step),
            "mean_return": str(mean_return),
            "max_return": str(max_return),
            "min_return": str(min_return),
        })
        with open(result_filename, "w") as f:
            json.dump(results_data, f, indent=4)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs[0].observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs[0].action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_weights = branch_weights.reshape(-1)

        # OPTS_TTPO: full-batch weighted advantage normalization
        if args.norm_adv:
            adv_mean = (b_advantages * b_weights).sum() / b_weights.sum()
            adv_var = ((b_advantages - adv_mean) ** 2 * b_weights).sum() / (
                b_weights.sum() - (b_weights**2).sum() / b_weights.sum()
            )
            b_advantages = (b_advantages - adv_mean) / (torch.sqrt(adv_var) + 1e-8)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                loss_norm = len(mb_inds)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                w = b_weights[mb_inds]

                # Policy loss (weighted by branch weights)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_loss = wagg(pg_loss_per_sample)

                # Value loss (weighted by branch weights)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * wagg(v_loss_max)
                else:
                    v_loss_per_sample = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * wagg(v_loss_per_sample)

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        session_steps = max(global_step - session_start_step, 1)
        sps = int(session_steps / (time.time() - start_time))
        print("SPS:", sps)
        print()
        writer.add_scalar("charts/SPS", sps, global_step)

        if (
            args.save_checkpoint
            and args.checkpoint_interval > 0
            and (iteration % args.checkpoint_interval == 0 or iteration == args.num_iterations)
        ):
            save_training_checkpoint(
                checkpoint_path,
                args,
                run_name,
                iteration,
                global_step,
                agent,
                optimizer,
                envs,
                next_obs,
                next_done,
            )

    for env in envs:
        env.close()
    writer.close()
