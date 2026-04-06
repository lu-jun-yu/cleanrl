# OPTS_TTPO (On-Policy Parallel Tree Search + Tree Trajectory Policy Optimization) for Atari
# Based on PPO implementation from CleanRL
import os
import random
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List

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

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 4096
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

    tau: float = 0.0
    """tau for the TUCT node selection"""
    max_search_per_tree: int = 4
    """maximum number of tree searches per environment per iteration"""
    c: float = 1.0
    """exploration coefficient for TUCT node selection"""

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

        # Capture RecordEpisodeStatistics values for state snapshot
        if terminated or truncated:
            # Episode ended - get the true total from info (RecordEpisodeStatistics already reset)
            if 'episode' in info:
                ep_r = info['episode']['r']
                ep_l = info['episode']['l']
                self._episode_return_snapshot = float(ep_r[0] if hasattr(ep_r, '__getitem__') else ep_r)
                self._episode_length_snapshot = int(ep_l[0] if hasattr(ep_l, '__getitem__') else ep_l)
            else:
                self._episode_return_snapshot = 0.0
                self._episode_length_snapshot = 0
        else:
            # Mid-episode - get current accumulated value from RecordEpisodeStatistics
            if self._record_stats_wrapper is not None:
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
        ale_state = self.ale.cloneState()

        # Save TimeLimit state
        timelimit_steps = None
        if self._timelimit_wrapper is not None:
            timelimit_steps = self._timelimit_wrapper._elapsed_steps

        # Save episode stats snapshot (captured in step() before any reset)
        record_stats = {
            'episode_returns': self._episode_return_snapshot,
            'episode_lengths': self._episode_length_snapshot,
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

        return (ale_state, timelimit_steps, record_stats, framestack_state, maxandskip_state, episodiclife_state)

    def restore_state(self, state):
        """Restore the environment to a previous state including all wrapper states."""
        ale_state, timelimit_steps, record_stats, framestack_state, maxandskip_state, episodiclife_state = state

        # Restore ALE state first
        self.ale.restoreState(ale_state)

        # Restore TimeLimit state
        if self._timelimit_wrapper is not None and timelimit_steps is not None:
            self._timelimit_wrapper._elapsed_steps = timelimit_steps

        # Restore RecordEpisodeStatistics state using array indexing
        if self._record_stats_wrapper is not None and record_stats is not None:
            self._record_stats_wrapper.episode_returns[0] = record_stats['episode_returns']
            self._record_stats_wrapper.episode_lengths[0] = record_stats['episode_lengths']
            # Also update our snapshot
            self._episode_return_snapshot = record_stats['episode_returns']
            self._episode_length_snapshot = record_stats['episode_lengths']

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


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
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


def compute_tree_gae(
    terminal_step: int,
    env_idx: int,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    parent_indices: torch.Tensor,
    advantages: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    next_value: float = 0.0,
):
    """
    Compute TreeGAE advantages from terminal node back to root.
    Synchronizes advantages for identical state-action pairs.

    Args:
        terminal_step: The step index of the terminal node
        env_idx: The environment index
        rewards, values, dones, parent_indices: Tree data tensors
        advantages: Tensor to store computed advantages (modified in-place)
        gamma, gae_lambda: GAE parameters
        next_value: Bootstrap value for non-terminal leaf nodes
    """
    current = terminal_step

    while True:
        # By this implementation's indexing: values[current] is the value of the
        # state at the current tree node (after its parent action).
        V_current = values[current, env_idx]

        # V_next uses the child node's stored value (same parent-state index semantics).
        children = (parent_indices[: terminal_step + 1, env_idx] == current).nonzero(as_tuple=True)[0].tolist()
        if len(children) == 0:
            V_next = 0.0 if dones[current, env_idx] else next_value
        else:
            V_next = values[children[0], env_idx]

        # Compute delta and advantage
        delta = rewards[current, env_idx] + gamma * V_next - V_current
        if len(children) == 0:
            advantages[current, env_idx] = delta
        else:
            # Branch node
            child_advs = torch.stack([advantages[child, env_idx] for child in children])
            mean_child_adv = child_advs.mean()
            advantages[current, env_idx] = delta + gamma * gae_lambda * mean_child_adv

        # Move to parent (parent < 0 means root node)
        parent = parent_indices[current, env_idx].item()
        if parent < 0:
            break
        current = parent


def compute_branch_weight(
    num_steps: int,
    parent_indices: torch.Tensor,
    state_branches: torch.Tensor,
    env_indices: List[int],
    root_branch_counts: List[dict],
) -> torch.Tensor:
    """
    Compute branch weight factors for specified environments.
    W_t = W_parent * state_branches[parent]

    For root nodes (parent < 0), the initial weight is the number of branches
    originating from the same root state (from root_branch_counts).

    Args:
        num_steps: Number of steps collected
        parent_indices: Parent indices tensor (num_steps, num_envs)
        state_branches: State branches tensor (num_steps, num_envs)
        env_indices: List of environment indices to compute weights for
        root_branch_counts: List of dicts mapping root_id -> branch_count for each env

    Returns:
        weights: (num_steps, len(env_indices)) tensor of branch weight factors
    """
    device = parent_indices.device
    n_envs = len(env_indices)
    env_t = torch.tensor(env_indices, device=device, dtype=torch.long)

    weights = torch.ones((num_steps, n_envs), device=device, dtype=torch.float32)
    for step in range(num_steps):
        p_steps = parent_indices[step, env_t]
        is_root = p_steps < 0

        for i in is_root.nonzero(as_tuple=True)[0].tolist():
            env_idx = env_indices[i]
            tree_root_id = p_steps[i].item()
            weights[step, i] = root_branch_counts[env_idx].get(tree_root_id, 1)

        if not is_root.all():
            valid_parents = p_steps[~is_root]
            valid_env_t = env_t[~is_root]
            valid_indices = torch.arange(n_envs, device=device)[~is_root]
            p_weights = weights[valid_parents, valid_indices]
            p_branches = state_branches[valid_parents, valid_env_t]
            weights[step, ~is_root] = p_weights * p_branches

    return weights


def select_next_states(
    terminated_envs: list[int],
    current_step: int,
    num_steps: int,
    advantages: torch.Tensor,
    parent_indices: torch.Tensor,
    tree_indices: torch.Tensor,
    skip_search: bool,
    search_count: list[dict],
    max_search: int,
    max_exploitations: list[dict],
    c: float = 1.0,
    gamma: float = 0.99,
    tau: float = 0.0,
) -> list[int]:
    """
    OPTS-TTPO node selection (aligned with verify_scaling_variance / select_next_states_v2).

    No return-threshold gating. TUCT uses raw discounted cumulative advantage along the path
    (not divided by remaining length); mean_exploitation = exploitation / (n - k) is used to
    filter trees whose chosen node is not above the pooled mean of recorded max mean exploitations.

    TUCT = exploitation - c * exploration, with exploration = (sibling_count - 1) * max_abs_exploitation.
    """
    selected = []
    n_steps = current_step + 1

    for env_idx in terminated_envs:
        env_tree_ids = torch.unique(tree_indices[:n_steps, env_idx]).tolist()
        num_env_trees = len(env_tree_ids)

        if skip_search:
            selected.append(-(num_env_trees + 1))
            continue

        best_mean_exp_val = float('-inf')
        best_step_overall = None
        best_tree_id = None
        best_depth = None
        best_path_len = None

        for tid in env_tree_ids:
            current_count = search_count[env_idx].get(tid, 0)
            if current_count >= max_search:
                continue

            tree_mask = tree_indices[:n_steps, env_idx] == tid
            tree_node_steps = tree_mask.nonzero(as_tuple=True)[0]
            tree_advs = advantages[tree_node_steps, env_idx].clone()
            tree_parents = parent_indices[tree_node_steps, env_idx]

            root_local_mask = tree_parents < 0
            root_steps = tree_node_steps[root_local_mask]
            root_advs = advantages[root_steps, env_idx]
            best_root_local = root_advs.argmax().item()
            current_node = root_steps[best_root_local].item()

            path = [current_node]
            while True:
                children_of_node = (parent_indices[:n_steps, env_idx] == current_node) & tree_mask
                children_steps = children_of_node.nonzero(as_tuple=True)[0]
                if len(children_steps) == 0:
                    break
                child_advs = advantages[children_steps, env_idx]
                best_child = child_advs.argmax().item()
                current_node = children_steps[best_child].item()
                path.append(current_node)

            path_t = torch.tensor(path, device=advantages.device)
            path_local_mask = torch.isin(tree_node_steps, path_t)
            path_advs = tree_advs[path_local_mask]
            path_steps = tree_node_steps[path_local_mask]

            n = len(path_advs)
            exploitation = torch.zeros_like(path_advs)
            mean_exploitation = torch.zeros_like(path_advs)
            discounted_sum = 0.0
            for k in range(n - 1, -1, -1):
                discounted_sum = -path_advs[k].item() + gamma * discounted_sum
                exploitation[k] = discounted_sum / ((n - k) ** tau)
                mean_exploitation[k] = exploitation[k] / (n - k)

            path_parents_vals = tree_parents[path_local_mask]
            sibling_counts = torch.zeros(len(path_steps), device=advantages.device)
            for i in range(len(path_steps)):
                sibling_counts[i] = (tree_parents == path_parents_vals[i]).sum()

            max_abs_exploitation = exploitation.abs().max().item()
            if max_abs_exploitation == 0:
                max_abs_exploitation = 1.0

            exploration = (sibling_counts - 1) * max_abs_exploitation
            tuct = exploitation - c * exploration

            max_path_idx = tuct.argmax().item()
            if tid not in max_exploitations[env_idx]:
                max_exploitations[env_idx][tid] = mean_exploitation[max_path_idx].item()

            max_exploitation_values = [v for d in max_exploitations for v in d.values() if v > 0]
            if len(max_exploitation_values) < 1:
                continue
            mean_max_exploitations = float(np.mean(max_exploitation_values))
            if mean_exploitation[max_path_idx] <= mean_max_exploitations:
                continue

            max_mean_exp_val = mean_exploitation[max_path_idx].item()

            if max_mean_exp_val > best_mean_exp_val:
                best_mean_exp_val = max_mean_exp_val
                best_step_overall = path_steps[max_path_idx].item()
                best_tree_id = tid
                best_depth = max_path_idx
                best_path_len = len(path)

        if best_step_overall is None:
            print(f"    New Tree: best_mean_exp={best_mean_exp_val:.4f}")
            selected.append(-(num_env_trees + 1))
        else:
            search_count[env_idx][best_tree_id] = search_count[env_idx].get(best_tree_id, 0) + 1
            print(
                f"    Tree Search: env_idx={env_idx}, tree_id={best_tree_id}, "
                f"mean_exp={best_mean_exp_val:.4f}, "
                f"search_count={search_count[env_idx][best_tree_id]}, "
                f"depth={best_depth} / {best_path_len}"
            )
            selected.append(best_step_overall)

    return selected


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    algorithm_name = f"{args.exp_name}_tau{args.tau}_s{args.max_search_per_tree}_20260406"
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

    # SyncVectorEnv for Agent init (cleanrl convention), envs list for actual training
    envs_vec = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    agent = Agent(envs_vec).to(device)
    envs_vec.close()
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

    for env_idx, env in enumerate(envs):
        obs_data, _ = env.reset(seed=args.seed + env_idx)
        next_obs[env_idx] = torch.Tensor(obs_data).to(device)
        root_states[env_idx] = [env.clone_state()]

    for iteration in range(1, args.num_iterations + 1):
        # Initialize episodic_returns for this iteration
        episodic_returns = []
        episodic_return_info = []  # (episodic_return, tid, step, env_idx)

        # Initialize max_episodic_return
        max_episodic_return = [float('-inf')] * args.num_envs

        # OPTS_TTPO: root_branch_counts maintained incrementally
        root_branch_counts = [{} for _ in range(args.num_envs)]

        # search count per tree (inherit from previous iteration for continuing envs)
        search_count = [{} for _ in range(args.num_envs)]

        # pooled mean-exploitation stats for tree filtering (verify_scaling_variance v2)
        max_exploitations = [{} for _ in range(args.num_envs)]

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for env_idx, env in enumerate(envs):
            if next_done[env_idx] == 1:
                obs_data, _ = env.reset()
                next_obs[env_idx] = torch.Tensor(obs_data).to(device)
            root_states[env_idx] = [env.clone_state()]
        next_done.zero_()

        current_parent = [-1] * args.num_envs
        next_done.zero_()
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
                    root_id = p
                    root_branch_counts[env_idx][root_id] = root_branch_counts[env_idx].get(root_id, 0) + 1

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
                    episodic_returns.append(episodic_return)
                    episodic_return_info.append((episodic_return, tree_indices[step, env_idx].item(), step, env_idx))
                    print(f"global_step={global_step}, episodic_return={episodic_return:.4f}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

                    if episodic_return > max_episodic_return[env_idx]:
                        max_episodic_return[env_idx] = episodic_return

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

                skip_search = step >= args.num_steps - 1

                selected = select_next_states(
                    terminated_envs=terminated_envs,
                    current_step=step,
                    num_steps=args.num_steps,
                    advantages=advantages,
                    parent_indices=parent_indices,
                    tree_indices=tree_indices,
                    skip_search=skip_search,
                    search_count=search_count,
                    max_search=args.max_search_per_tree,
                    max_exploitations=max_exploitations,
                    c=args.c,
                    gamma=args.gamma,
                    tau=args.tau,
                )

                # TUCT selection and state restoration
                for i, env_idx in enumerate(terminated_envs):
                    if selected[i] < 0:
                        # Variance is stable, start a new tree
                        obs_data, _ = envs[env_idx].reset()
                        next_obs[env_idx] = torch.Tensor(obs_data).to(device)
                        root_states[env_idx].insert(0, envs[env_idx].clone_state())
                        current_parent[env_idx] = -len(root_states[env_idx])
                        max_episodic_return[env_idx] = float('-inf')
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

        # Compute branch_weight for all environments
        branch_weights = compute_branch_weight(
            num_steps=args.num_steps,
            parent_indices=parent_indices,
            state_branches=state_branches,
            env_indices=list(range(args.num_envs)),
            root_branch_counts=root_branch_counts,
        )

        # Compute tree-weighted aggregated returns
        if episodic_return_info:
            tid_groups = {}  # (env_idx, tid) -> [(return, step)]
            for ep_return, tid, ep_step, ei in episodic_return_info:
                key = (ei, tid)
                if key not in tid_groups:
                    tid_groups[key] = []
                tid_groups[key].append((ep_return, ep_step))

            aggregated_returns = []
            for (ei, tid), entries in tid_groups.items():
                weighted_sum = 0.0
                weight_sum = 0.0
                for ep_return, ep_step in entries:
                    w = branch_weights[ep_step, ei].item()
                    weighted_sum += ep_return / w
                    weight_sum += 1.0 / w
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
        result_filename = f"{folder_name}/{args.env_id}_{args.seed}.json"
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_weights = b_weights[mb_inds]
                mb_weights_sum = (1.0 / mb_weights).sum()
                if args.norm_adv:
                    mb_advantages_mean = (mb_advantages / mb_weights).sum() / mb_weights_sum
                    mb_advantages_var = ((mb_advantages - mb_advantages_mean) ** 2 / mb_weights).sum() / mb_weights_sum
                    mb_advantages_std = torch.sqrt(mb_advantages_var)
                    mb_advantages = (mb_advantages - mb_advantages_mean) / (mb_advantages_std + 1e-8)

                # Policy loss (weighted by branch factors)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_loss = (pg_loss_per_sample / mb_weights).sum() / mb_weights_sum

                # Value loss (weighted by branch factors)
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
                    v_loss = 0.5 * (v_loss_max / mb_weights).sum() / mb_weights_sum
                else:
                    v_loss_per_sample = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (v_loss_per_sample / mb_weights).sum() / mb_weights_sum

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
        print("SPS:", int(global_step / (time.time() - start_time)))
        print()
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    for env in envs:
        env.close()
    writer.close()
