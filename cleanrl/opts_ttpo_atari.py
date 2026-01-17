# OPTS_TTPO (On-Policy Parallel Tree Search + Tree Trajectory Policy Optimization) for Atari
# Based on PPO implementation from CleanRL
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

    # OPTS_TTPO specific arguments
    root_tuct: float = 0.5
    """the baseline TUCT value for root node selection"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class AtariStateSnapshotWrapper(gym.Wrapper):
    """Wrapper to support state snapshots for Atari environments using ALE."""

    def __init__(self, env):
        super().__init__(env)
        self.ale = self.unwrapped.ale
        # Find TimeLimit and RecordEpisodeStatistics wrappers
        self._timelimit_wrapper = None
        self._record_stats_wrapper = None
        current = env
        while current is not None:
            if hasattr(current, '_elapsed_steps'):  # TimeLimit wrapper
                self._timelimit_wrapper = current
            if hasattr(current, 'episode_returns'):  # RecordEpisodeStatistics wrapper
                self._record_stats_wrapper = current
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
        """Clone the current environment state including wrapper states."""
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

        return (ale_state, timelimit_steps, record_stats)

    def restore_state(self, state):
        """Restore the environment to a previous state including wrapper states."""
        ale_state, timelimit_steps, record_stats = state
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

        return self.ale.getScreenRGB() if hasattr(self.ale, 'getScreenRGB') else None


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
        self.actor = layer_init(nn.Linear(512, envs[0].single_action_space.n), std=0.01)
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
    children_indices: List[List[List[int]]],
    advantages: torch.Tensor,
    tid: torch.Tensor,
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
        children_indices: List of children indices for each node
        advantages: Tensor to store computed advantages (modified in-place)
        tid: Trajectory ID for each node
        gamma, gae_lambda: GAE parameters
        next_value: Bootstrap value for non-terminal leaf nodes
    """
    current = terminal_step

    while True:
        # Get current state value V(s_t)
        V_current = values[current, env_idx]

        # Compute V(next_state) from children or terminal
        children = children_indices[current][env_idx]
        if len(children) == 0:
            V_next = 0.0 if dones[current, env_idx] else next_value
        else:
            V_next = values[children[0], env_idx]

        # Compute delta and advantage
        delta = rewards[current, env_idx] + gamma * V_next - V_current
        if len(children) == 0:
            current_adv = delta
        else:
            current_adv = delta + gamma * gae_lambda * advantages[children[0], env_idx]

        # Sync advantages for identical state-action pairs
        parent = parent_indices[current, env_idx].item()
        if parent != -1:
            siblings = children_indices[parent][env_idx]
            n = len(siblings)
            old_avg = advantages[siblings[0], env_idx]
            new_avg = (old_avg * (n - 1) + current_adv) / n
            for s in siblings:
                advantages[s, env_idx] = new_avg
        else:
            current_tid = int(tid[current, env_idx].item())
            first_layer_siblings = []
            for s in range(terminal_step + 1):
                if (parent_indices[s, env_idx].item() == -1 and
                    int(tid[s, env_idx].item()) == current_tid):
                    first_layer_siblings.append(s)
            n = len(first_layer_siblings)
            old_avg = advantages[first_layer_siblings[0], env_idx]
            new_avg = (old_avg * (n - 1) + current_adv) / n
            for s in first_layer_siblings:
                advantages[s, env_idx] = new_avg

        # Move to parent
        if parent == -1:
            break
        current = parent


def compute_branch_weight_factors(
    num_steps: int,
    parent_indices: torch.Tensor,
    state_branches: torch.Tensor,
    tid: torch.Tensor,
    init_weights: List[Dict[int, int]],
    env_indices: List[int],
) -> torch.Tensor:
    """
    Compute branch weight factors for specified environments.
    W_t = init_weights[tid] if parent=-1, else W_parent * state_branches[parent]

    Args:
        num_steps: Number of steps collected
        parent_indices: Parent indices tensor (num_steps, num_envs)
        state_branches: State branches tensor (num_steps, num_envs)
        tid: Trajectory ID tensor (num_steps, num_envs)
        init_weights: List of dicts mapping tid to initial weight for each env
        env_indices: List of environment indices to compute weights for

    Returns:
        weights: (num_steps, len(env_indices)) tensor of branch weight factors
    """
    device = parent_indices.device
    num_target_envs = len(env_indices)
    weights = torch.zeros((num_steps, num_target_envs), device=device, dtype=torch.float32)

    # Convert env_indices to tensor for vectorized operations
    env_tensor = torch.tensor(env_indices, device=device, dtype=torch.long)

    for step in range(num_steps):
        # Get parent indices for target environments
        parent_steps = parent_indices[step, env_tensor]  # (num_target_envs,)

        # Mask for first layer nodes (parent == -1)
        first_layer = parent_steps == -1

        # First layer nodes: use init_weights[tid]
        if first_layer.any():
            for i in range(num_target_envs):
                if first_layer[i]:
                    env_idx = env_indices[i]
                    tid_val = int(tid[step, env_idx].item())
                    weights[step, i] = float(init_weights[env_idx].get(tid_val, 1))

        # For other nodes, weight = parent_weight * parent_state_branches
        if (~first_layer).any():
            valid_parents = parent_steps[~first_layer]
            valid_env_tensor = env_tensor[~first_layer]
            valid_indices = torch.arange(num_target_envs, device=device)[~first_layer]

            # Gather parent weights and branches
            parent_weights = weights[valid_parents, valid_indices]
            parent_branches = state_branches[valid_parents, valid_env_tensor]

            # Compute weights
            weights[step, ~first_layer] = parent_weights * parent_branches

    return weights


def select_next_action(
    env_idx: int,
    current_step: int,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    tid: torch.Tensor,
    tree_branches: Dict[int, int],
    N_total: int,
    root_tuct: float,
    dones: torch.Tensor,  # 终止标志，mask 掉 done=True 的位置
    parent_indices: torch.Tensor = None,  # 聚合参数（注释调用处可恢复原版）
    children_indices: List[List[List[int]]] = None,  # 聚合参数（注释调用处可恢复原版）
) -> int:
    """
    Select the action with highest TUCT value using vectorized operations.
    TUCT = (A_t / W_t) * sqrt(log(N_total + 1)) / N_subtree

    Note: Each index t represents a state-action pair (s_parent, a_t).
    This function selects which action to repeat from its parent state.

    Args:
        env_idx: The environment index
        current_step: Current step index (select from actions 0 to current_step)
        advantages: Advantages tensor (num_steps, num_envs)
        weights: Branch weight factors (current_step+1, 1) for this env
        tid: Trajectory ID tensor (num_steps, num_envs)
        tree_branches: Dictionary mapping tid to branch count
        N_total: Total number of terminal trajectories
        root_tuct: Baseline TUCT value for selecting root state

    Returns:
        Selected action index (-1 for root state, meaning start new trajectory)
    """
    if current_step < 0:
        return -1

    adv = advantages[:current_step + 1, env_idx]
    w = weights[:current_step + 1, 0]
    tid_values = tid[:current_step + 1, env_idx]
    done_mask = dones[:current_step + 1, env_idx] == True

    N_subtree = torch.tensor(
        [tree_branches.get(int(t.item()), 1) for t in tid_values],
        dtype=torch.float32,
        device=adv.device
    )

    exploitation = adv / w
    exploration = torch.sqrt(torch.log(torch.tensor(N_total + 1, dtype=torch.float32, device=adv.device))) / N_subtree
    tuct = exploitation * exploration

    # === TUCT Aggregation（注释 if 分支并取消注释 else 分支可恢复原版）===
    if parent_indices is not None and children_indices is not None:
        # Aggregate TUCT for identical state-action pairs (skip done=True)
        aggregated_tuct = {}
        for idx in range(current_step + 1):
            if done_mask[idx]:
                continue
            parent = int(parent_indices[idx, env_idx].item())
            if parent != -1:
                representative = children_indices[parent][env_idx][0]
            else:
                current_tid = int(tid_values[idx].item())
                for s in range(current_step + 1):
                    if (int(parent_indices[s, env_idx].item()) == -1 and
                        int(tid_values[s].item()) == current_tid):
                        representative = s
                        break
            if representative not in aggregated_tuct:
                aggregated_tuct[representative] = 0.0
            aggregated_tuct[representative] += tuct[idx].item()
        best_action_idx = max(aggregated_tuct, key=aggregated_tuct.get)
        best_tuct = aggregated_tuct[best_action_idx]
    else:
        tuct[done_mask] = float('-inf')
        best_action_idx = tuct.argmax().item()
        best_tuct = tuct[best_action_idx].item()

    root_tuct_value = max(adv.mean().item(), root_tuct)

    if root_tuct_value > best_tuct:
        return -1
    else:
        return best_action_idx


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs[0].observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs[0].action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # OPTS_TTPO: Tree structure tensors
    parent_indices = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    state_branches = torch.ones((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    tid = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    advantages = torch.zeros((args.num_steps, args.num_envs)).to(device)
    returns = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # OPTS_TTPO: Non-tensor data
    env_states = [[None for _ in range(args.num_envs)] for _ in range(args.num_steps)]
    children_indices = [[[] for _ in range(args.num_envs)] for _ in range(args.num_steps)]
    tree_branches = [{0: 1} for _ in range(args.num_envs)]  # Initialize with tid=0, branches=1
    init_weights = [{0: 1} for _ in range(args.num_envs)]  # Initial weights for first layer nodes
    next_tid = [1] * args.num_envs  # Next available tid for each env

    # OPTS_TTPO: Root node data
    root_states = [None] * args.num_envs

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.zeros((args.num_envs,) + envs[0].observation_space.shape).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Reset all environments to new random initial states
        for env_idx, env in enumerate(envs):
            obs_data, _ = env.reset()
            next_obs[env_idx] = torch.Tensor(obs_data).to(device)
            root_states[env_idx] = env.clone_state()

        # Reset tree structures for new iteration
        current_parent = [-1] * args.num_envs
        pending_action = [None] * args.num_envs
        pending_logprob = [None] * args.num_envs
        N_total = [0] * args.num_envs
        current_tid = [0] * args.num_envs
        tree_branches = [{0: 1} for _ in range(args.num_envs)]
        init_weights = [{0: 1} for _ in range(args.num_envs)]
        next_tid = [1] * args.num_envs
        next_done.zero_()

        # Reset state_branches to all ones
        state_branches.fill_(1)

        # Reset advantages, tid, parent_indices
        advantages.zero_()
        tid.zero_()
        parent_indices.zero_()

        # Reset children_indices
        for step_idx in range(args.num_steps):
            for env_idx in range(args.num_envs):
                children_indices[step_idx][env_idx] = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # Save current obs and done (like PPO)
            obs[step] = next_obs
            dones[step] = next_done

            # Sample actions
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            # Override with pending actions if any
            for env_idx in range(args.num_envs):
                if pending_action[env_idx] is not None:
                    action[env_idx] = pending_action[env_idx]
                    logprob[env_idx] = pending_logprob[env_idx]
                    pending_action[env_idx] = None
                    pending_logprob[env_idx] = None

            actions[step] = action
            logprobs[step] = logprob

            # Execute actions in all environments
            next_obs_list = []
            rewards_list = []
            next_done_list = []

            for env_idx, env in enumerate(envs):
                next_o, r, terminated, truncated, info = env.step(action[env_idx].cpu().item())
                next_obs_list.append(next_o)
                rewards_list.append(r)
                next_done_list.append(terminated or truncated)

                # Save state snapshot (after action)
                env_states[step][env_idx] = env.clone_state()

                # Log episode info
                if (terminated or truncated) and "episode" in info:
                    ep_r = info['episode']['r']
                    ep_l = info['episode']['l']
                    # Extract scalar value from array if needed
                    episodic_return = float(ep_r[0] if hasattr(ep_r, '__getitem__') else ep_r)
                    episodic_length = int(ep_l[0] if hasattr(ep_l, '__getitem__') else ep_l)
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

            # Convert to tensors
            next_obs = torch.stack([torch.Tensor(o) for o in next_obs_list]).to(device)
            rewards[step] = torch.tensor(rewards_list).to(device)
            next_done = torch.tensor(next_done_list, dtype=torch.float32).to(device)

            # Update dones[step] to reflect terminal status after this step's action
            dones[step] = next_done

            # Save parent_indices and tid
            for env_idx in range(args.num_envs):
                parent_indices[step, env_idx] = current_parent[env_idx]

                if current_parent[env_idx] == -1:
                    tid[step, env_idx] = current_tid[env_idx]
                else:
                    tid[step, env_idx] = tid[current_parent[env_idx], env_idx]
                    children_indices[current_parent[env_idx]][env_idx].append(step)

            # Handle terminated environments
            terminated_envs = [i for i in range(args.num_envs) if next_done_list[i]]

            if len(terminated_envs) > 0:
                # TreeGAE update
                for env_idx in terminated_envs:
                    N_total[env_idx] += 1
                    compute_tree_gae(
                        terminal_step=step,
                        env_idx=env_idx,
                        rewards=rewards,
                        values=values,
                        dones=dones,
                        parent_indices=parent_indices,
                        children_indices=children_indices,
                        advantages=advantages,
                        tid=tid,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                    )

                # Compute branch_weight_factors for all terminated envs at once
                weights = compute_branch_weight_factors(
                    num_steps=step + 1,
                    parent_indices=parent_indices,
                    state_branches=state_branches,
                    tid=tid,
                    init_weights=init_weights,
                    env_indices=terminated_envs,
                )

                # TUCT selection and state restoration
                for i, env_idx in enumerate(terminated_envs):
                    # Select next action using pre-computed weights
                    selected = select_next_action(
                        env_idx=env_idx,
                        current_step=step,
                        advantages=advantages,
                        weights=weights[:, i:i+1],
                        tid=tid,
                        tree_branches=tree_branches[env_idx],
                        N_total=N_total[env_idx],
                        root_tuct=args.root_tuct,
                        dones=dones,
                        parent_indices=parent_indices,  # 聚合参数（注释可恢复原版）
                        children_indices=children_indices,  # 聚合参数（注释可恢复原版）
                    )

                    # State restoration
                    if selected == -1:
                        # Selected root
                        envs[env_idx].restore_state(root_states[env_idx])
                        next_obs[env_idx] = obs[0, env_idx]
                        current_parent[env_idx] = -1
                        pending_action[env_idx] = None

                        # Create new trajectory
                        current_tid[env_idx] = next_tid[env_idx]
                        tree_branches[env_idx][current_tid[env_idx]] = 1
                        init_weights[env_idx][current_tid[env_idx]] = 1
                        next_tid[env_idx] += 1
                    else:
                        # Selected action - restore to parent state and repeat the action
                        parent = parent_indices[selected, env_idx].item()
                        if parent == -1:
                            envs[env_idx].restore_state(root_states[env_idx])
                            next_obs[env_idx] = obs[0, env_idx]
                            init_weights[env_idx][current_tid[env_idx]] += 1
                        else:
                            envs[env_idx].restore_state(env_states[parent][env_idx])
                            next_obs[env_idx] = obs[selected, env_idx]
                            state_branches[parent, env_idx] += 1

                        current_parent[env_idx] = parent
                        pending_action[env_idx] = actions[selected, env_idx]
                        pending_logprob[env_idx] = logprobs[selected, env_idx]
                        current_tid[env_idx] = int(tid[selected, env_idx].item())
                        tree_branches[env_idx][current_tid[env_idx]] += 1

            # Update non-terminated environments
            for env_idx in range(args.num_envs):
                if not next_done_list[env_idx]:
                    current_parent[env_idx] = step

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
                    children_indices=children_indices,
                    advantages=advantages,
                    tid=tid,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    next_value=next_value[env_idx].item(),
                )

        # Compute returns: returns[t] = A(s_t, a_t) + V(s_t)
        returns = advantages + values

        # Compute branch_weight_factors for all environments
        branch_weights = compute_branch_weight_factors(
            num_steps=args.num_steps,
            parent_indices=parent_indices,
            state_branches=state_branches,
            tid=tid,
            init_weights=init_weights,
            env_indices=list(range(args.num_envs)),
        )

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
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)

                # Aggregate loss with branch_weight_factor correction
                mb_weights = b_weights[mb_inds]
                pg_loss = (pg_loss_per_sample / mb_weights).sum() / (1.0 / mb_weights).sum()

                # Value loss
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
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    for env in envs:
        env.close()
    writer.close()
