# OPTS_TTPO (On-Policy Parallel Tree Search + Tree Trajectory Policy Optimization) for Continuous Action
# Based on PPO implementation from CleanRL
import os
import random
import time
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Any, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
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


class MuJoCoStateSnapshotWrapper(gym.Wrapper):
    """Wrapper to support state snapshots for MuJoCo environments."""

    def __init__(self, env):
        super().__init__(env)
        # Find TimeLimit and RecordEpisodeStatistics wrappers
        self._timelimit_wrapper = None
        self._record_stats_wrapper = None
        self._normalize_obs_wrapper = None
        self._normalize_reward_wrapper = None
        current = env
        while current is not None:
            if hasattr(current, '_elapsed_steps'):  # TimeLimit wrapper
                self._timelimit_wrapper = current
            if hasattr(current, 'episode_returns'):  # RecordEpisodeStatistics wrapper
                self._record_stats_wrapper = current
            # Normalization wrappers keep internal running statistics; they must be snapshotted too
            if isinstance(current, gym.wrappers.NormalizeObservation):
                self._normalize_obs_wrapper = current
            if isinstance(current, gym.wrappers.NormalizeReward):
                self._normalize_reward_wrapper = current
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
        """Clone the current environment state including wrapper states.
        
        Copies all key mjData fields to ensure complete state restoration.
        Also saves derived quantities (xpos, site_xpos, etc.) which are needed
        because Gymnasium MuJoCo envs return observations using "stale" derived
        quantities computed before qpos/qvel integration in mj_step.
        """
        env = self.unwrapped
        data = env.data
        
        # Copy all key mjData fields (covers physics state + derived quantities)
        mj_state = {
            'qpos': data.qpos.copy(),
            'qvel': data.qvel.copy(),
            'time': float(data.time),
        }
        
        # Optional input fields that may exist
        optional_fields = [
            'act', 'qacc', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied',
            'ctrl', 'qfrc_actuator', 'qfrc_bias', 'qfrc_constraint',
            'qacc_smooth', 'qfrc_inverse',
            # Contacts and derived quantities (important for reward calculation)
            'cfrc_int', 'cfrc_ext',
        ]
        for field in optional_fields:
            if hasattr(data, field):
                arr = getattr(data, field)
                if arr is not None and hasattr(arr, 'copy') and arr.size > 0:
                    mj_state[field] = arr.copy()
        
        # Mocap bodies (for Reacher-like envs)
        if hasattr(data, 'mocap_pos') and data.mocap_pos is not None and data.mocap_pos.size > 0:
            mj_state['mocap_pos'] = data.mocap_pos.copy()
        if hasattr(data, 'mocap_quat') and data.mocap_quat is not None and data.mocap_quat.size > 0:
            mj_state['mocap_quat'] = data.mocap_quat.copy()
        
        # Save derived quantities needed for observation calculation
        # These are computed by mj_forward/mj_step1 based on qpos/qvel BEFORE integration
        # Gymnasium envs use these "stale" values in _get_obs() after step()
        derived_fields = [
            'xpos', 'xquat', 'xmat',           # Body positions/orientations
            'xipos', 'ximat',                   # Body inertia positions
            'site_xpos', 'site_xmat',           # Site positions (used by Reacher)
            'subtree_com',                      # Subtree center of mass (used by get_body_com)
            'cinert', 'cvel', 'cacc',           # Composite body inertia/velocity/acceleration
            'cdof', 'cdof_dot',                 # DoF-related derivatives (Humanoid)
        ]
        derived_state = {}
        for field in derived_fields:
            if hasattr(data, field):
                arr = getattr(data, field)
                if arr is not None and hasattr(arr, 'copy') and arr.size > 0:
                    derived_state[field] = arr.copy()
        
        # Save goal for goal-conditioned envs like Reacher-v4
        goal = None
        try:
            if hasattr(env, "goal") and env.goal is not None:
                goal = np.array(env.goal, copy=True)
        except Exception:
            goal = None

        # Save RNG state (important for envs with per-episode random goals)
        rng_state = None
        try:
            if hasattr(env, "np_random") and env.np_random is not None:
                rng_state = env.np_random.bit_generator.state
        except Exception:
            rng_state = None
        
        # Save environment-specific Python attributes that affect reward/obs
        env_attrs = {}
        try:
            for attr_name in ['_last_x_position', '_last_position', '_init_obs']:
                if hasattr(env, attr_name):
                    val = getattr(env, attr_name)
                    if val is not None:
                        if hasattr(val, 'copy'):
                            env_attrs[attr_name] = val.copy()
                        else:
                            env_attrs[attr_name] = val
        except Exception:
            pass

        # Save TimeLimit state
        timelimit_steps = None
        if self._timelimit_wrapper is not None:
            timelimit_steps = self._timelimit_wrapper._elapsed_steps

        # Save episode stats snapshot (captured in step() before any reset)
        record_stats = {
            'episode_returns': self._episode_return_snapshot,
            'episode_lengths': self._episode_length_snapshot,
        }

        # Save normalization wrapper states (RunningMeanStd + discounted return accumulator)
        norm_obs_state = None
        if self._normalize_obs_wrapper is not None and hasattr(self._normalize_obs_wrapper, "obs_rms"):
            obs_rms = self._normalize_obs_wrapper.obs_rms
            norm_obs_state = {
                "mean": np.array(obs_rms.mean, copy=True),
                "var": np.array(obs_rms.var, copy=True),
                "count": float(obs_rms.count),
            }
        norm_reward_state = None
        if self._normalize_reward_wrapper is not None:
            state = {}
            if hasattr(self._normalize_reward_wrapper, "return_rms") and self._normalize_reward_wrapper.return_rms is not None:
                rr = self._normalize_reward_wrapper.return_rms
                state["return_rms"] = {
                    "mean": np.array(rr.mean, copy=True),
                    "var": np.array(rr.var, copy=True),
                    "count": float(rr.count),
                }
            if hasattr(self._normalize_reward_wrapper, "returns"):
                # returns can be scalar or array depending on wrapper version
                rets = self._normalize_reward_wrapper.returns
                state["returns"] = np.array(rets, copy=True) if hasattr(rets, "__array__") else float(rets)
            norm_reward_state = state if len(state) > 0 else None

        return (mj_state, goal, rng_state, timelimit_steps, record_stats, norm_obs_state, norm_reward_state, env_attrs, derived_state)

    def restore_state(self, state):
        """Restore the environment to a previous state including wrapper states.
        
        Restores all key mjData fields for complete state restoration.
        Key fix: Restore all fields BEFORE calling mj_forward to ensure derived
        quantities (site positions, contact forces, etc.) are computed correctly.
        Then restore the saved derived quantities to match Gymnasium's behavior
        where observations use "stale" derived values from before integration.
        """
        import mujoco
        
        mj_state, goal, rng_state, timelimit_steps, record_stats, norm_obs_state, norm_reward_state, env_attrs, derived_state = state
        
        env = self.unwrapped
        data = env.data
        model = env.model
        
        # Step 1: Restore all mjData fields FIRST (before mj_forward)
        # Restore qpos and qvel directly (not through set_state yet)
        data.qpos[:] = mj_state['qpos']
        data.qvel[:] = mj_state['qvel']
        data.time = mj_state['time']
        
        # Restore all other mjData fields (including mocap_pos, ctrl, etc.)
        for field, value in mj_state.items():
            if field in ('qpos', 'qvel', 'time'):
                continue  # Already restored
            if hasattr(data, field):
                target = getattr(data, field)
                if target is not None and hasattr(target, '__setitem__'):
                    try:
                        target[:] = value
                    except Exception:
                        pass
        
        # Restore goal for goal-conditioned envs like Reacher-v4
        # Must be done BEFORE mj_forward for environments that use goal in observations
        try:
            if goal is not None and hasattr(env, "goal"):
                env.goal = np.array(goal, copy=True)
        except Exception:
            pass
        
        # Step 2: Call mj_forward to recompute all derived quantities
        # This ensures site_xpos, xipos, subtree_com, cfrc_int, cfrc_ext, etc. are correct
        mujoco.mj_forward(model, data)
        
        # Step 3: Restore saved derived quantities AFTER mj_forward
        # This is needed because Gymnasium MuJoCo envs return observations using
        # "stale" derived quantities computed BEFORE qpos/qvel integration in mj_step.
        # Without this, the observation immediately after restore won't match the
        # observation at save time (though subsequent steps will be deterministic).
        if derived_state:
            for field, value in derived_state.items():
                if hasattr(data, field):
                    target = getattr(data, field)
                    if target is not None and hasattr(target, '__setitem__'):
                        try:
                            target[:] = value
                        except Exception:
                            pass
        
        # Restore environment-specific Python attributes
        try:
            if env_attrs:
                for attr_name, val in env_attrs.items():
                    if hasattr(env, attr_name):
                        if hasattr(val, 'copy'):
                            setattr(env, attr_name, val.copy())
                        else:
                            setattr(env, attr_name, val)
        except Exception:
            pass

        # Restore RNG state
        try:
            if rng_state is not None and hasattr(self.unwrapped, "np_random") and self.unwrapped.np_random is not None:
                self.unwrapped.np_random.bit_generator.state = rng_state
        except Exception:
            pass

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

        # Restore normalization statistics
        if self._normalize_obs_wrapper is not None and norm_obs_state is not None and hasattr(self._normalize_obs_wrapper, "obs_rms"):
            obs_rms = self._normalize_obs_wrapper.obs_rms
            obs_rms.mean = np.array(norm_obs_state["mean"], copy=True)
            obs_rms.var = np.array(norm_obs_state["var"], copy=True)
            obs_rms.count = float(norm_obs_state["count"])
        if self._normalize_reward_wrapper is not None and norm_reward_state is not None:
            if "return_rms" in norm_reward_state and hasattr(self._normalize_reward_wrapper, "return_rms") and self._normalize_reward_wrapper.return_rms is not None:
                rr = self._normalize_reward_wrapper.return_rms
                rr.mean = np.array(norm_reward_state["return_rms"]["mean"], copy=True)
                rr.var = np.array(norm_reward_state["return_rms"]["var"], copy=True)
                rr.count = float(norm_reward_state["return_rms"]["count"])
            if "returns" in norm_reward_state and hasattr(self._normalize_reward_wrapper, "returns"):
                self._normalize_reward_wrapper.returns = norm_reward_state["returns"]


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = MuJoCoStateSnapshotWrapper(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


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
    advantages: torch.Tensor,
    parent_indices: torch.Tensor,
    tree_indices: torch.Tensor,
    search_count: list[dict],
    max_search: int,
    max_otrc_scores: list[dict],
    skip_init_search: list[bool],
    gamma: float = 0.99,
    tau: float = 0.7,
) -> list[int]:
    """
    OPTS-TTPO node selection, vectorized over all terminated envs' trees at once (flat id = step*E + e_local).
    Trees are registered first, then gated in one order-independent pass; selection stays per-env.
    """
    selected = []
    n_steps = current_step + 1
    device = advantages.device
    dtype = advantages.dtype
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    E = len(terminated_envs)
    env_arr = torch.tensor(terminated_envs, device=device, dtype=torch.long)
    sub_par = parent_indices[:n_steps, env_arr]
    sub_advs = advantages[:n_steps, env_arr]
    sub_trees = tree_indices[:n_steps, env_arr]
    N = n_steps * E

    # flat id = step*E + e_local; a child and its parent share e_local, so envs never mix.
    step_grid = torch.arange(n_steps, device=device).unsqueeze(1).expand(n_steps, E)
    e_grid = torch.arange(E, device=device).unsqueeze(0).expand(n_steps, E)

    is_child = sub_par >= 0
    child_nodes = (step_grid * E + e_grid)[is_child]
    parent_of_child = (sub_par * E + e_grid)[is_child]
    child_adv = sub_advs[is_child]
    group_max = torch.full((N,), float("-inf"), device=device, dtype=dtype)
    group_max.scatter_reduce_(0, parent_of_child, child_adv, reduce="amax", include_self=True)
    qual = child_adv == group_max[parent_of_child]
    best_child = torch.full((N,), -1, device=device, dtype=torch.long)
    best_child.scatter_reduce_(0, parent_of_child[qual], child_nodes[qual], reduce="amin", include_self=False)

    active_tids, roots, tree_e_local = [], [], []
    env_num_trees = [0] * E
    for e_local, env_idx in enumerate(terminated_envs):
        col_trees = sub_trees[:, e_local]
        col_parents = sub_par[:, e_local]
        col_advs = sub_advs[:, e_local]
        env_tree_ids = torch.unique(col_trees).tolist()
        env_num_trees[e_local] = len(env_tree_ids)
        for tid in env_tree_ids:
            if (skip_init_search[env_idx] and tid == -1) or search_count[env_idx].get(tid, 0) >= max_search:
                continue
            root_steps = ((col_trees == tid) & (col_parents < 0)).nonzero(as_tuple=True)[0]
            active_tids.append(tid)
            roots.append(root_steps[col_advs[root_steps].argmax()].item() * E + e_local)
            tree_e_local.append(e_local)

    T = len(active_tids)
    eligible = torch.zeros(T, device=device, dtype=torch.bool)
    e_local_arr = torch.tensor(tree_e_local, device=device, dtype=torch.long)
    if T:
        cur = torch.tensor(roots, device=device, dtype=torch.long)

        active = torch.ones(T, device=device, dtype=torch.bool)
        path_cols, mask_cols = [], []
        while bool(active.any()):
            path_cols.append(cur)
            mask_cols.append(active.clone())

            child = best_child[cur]
            active = active & (child >= 0)
            cur = torch.where(active, child, cur)

        path_idx = torch.stack(path_cols, dim=1)
        path_mask = torch.stack(mask_cols, dim=1)
        row = torch.arange(T, device=device)
        path_idx = torch.cat([path_idx, cur[:, None]], dim=1)
        path_mask = torch.cat([path_mask, torch.zeros((T, 1), device=device, dtype=torch.bool)], dim=1)
        virtual_pos = (~path_mask).int().argmax(dim=1)
        path_mask[row, virtual_pos] = True
        n_t = path_mask.sum(dim=1)

        path_adv = sub_advs.reshape(-1)[path_idx].masked_fill(~path_mask, 0.0)
        path_adv[row, virtual_pos] = 0.0
        otrc_score = torch.zeros_like(path_adv)
        discounted = torch.zeros(T, device=device, dtype=dtype)
        for k in range(path_idx.shape[1] - 1, -1, -1):
            m = path_mask[:, k].to(dtype)
            discounted = (-path_adv[:, k] + gamma * discounted) * m + discounted * (1 - m)
            divisor = torch.where(path_mask[:, k], n_t - k, 1).to(dtype) ** tau
            otrc_score[:, k] = torch.where(path_mask[:, k], discounted / divisor, torch.zeros_like(discounted))

        max_pos = torch.where(path_mask, otrc_score, neg_inf).argmax(dim=1)
        max_otrc_score = otrc_score[row, max_pos]
        best_step = path_idx[row, max_pos] // E

        for i, tid in enumerate(active_tids):
            max_otrc_scores[terminated_envs[tree_e_local[i]]].setdefault(tid, float(max_otrc_score[i].item()))

        pool = [v for d in max_otrc_scores for v in d.values()]
        if len(pool) > 1:
            eligible = max_otrc_score > float(np.mean(pool))

    for e_local, env_idx in enumerate(terminated_envs):
        mask = (e_local_arr == e_local) & eligible
        if not bool(mask.any()):
            print(f"    New Tree: best_otrc_score={float('-inf'):.4f}")
            selected.append(-(env_num_trees[e_local] + 1))
            continue

        best_t = int(torch.where(mask, max_otrc_score, neg_inf).argmax().item())
        best_tid = active_tids[best_t]
        search_count[env_idx][best_tid] = search_count[env_idx].get(best_tid, 0) + 1
        print(
            f"    Tree Search: env_idx={env_idx}, tree_id={best_tid}, "
            f"otrc_score={max_otrc_score[best_t].item():.4f}, "
            f"search_count={search_count[env_idx][best_tid]}, "
            f"depth={int(max_pos[best_t].item())} / {int(n_t[best_t].item())}"
        )
        selected.append(int(best_step[best_t].item()))

    return selected


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    algorithm_name = f"{args.exp_name}_tau{args.tau}_s{args.max_search_per_tree}_20260605"
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
    envs = [make_env(args.env_id, i, args.capture_video, run_name, args.gamma)() for i in range(args.num_envs)]
    assert isinstance(envs[0].action_space, gym.spaces.Box), "only continuous action space is supported"

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

        # pooled mean-otrc_score stats for tree filtering (verify_scaling_variance v2)
        max_otrc_scores = [{} for _ in range(args.num_envs)]

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

                if step < args.num_steps - 1:
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_weights = b_weights[mb_inds]
                w = 1.0 / mb_weights
                w_sum = w.sum()
                if args.norm_adv:
                    mb_advantages_mean = (mb_advantages * w).sum() / w_sum
                    mb_advantages_var = ((mb_advantages - mb_advantages_mean) ** 2 * w).sum() / (w_sum - 1)
                    mb_advantages_std = torch.sqrt(mb_advantages_var)
                    mb_advantages = (mb_advantages - mb_advantages_mean) / (mb_advantages_std + 1e-8)

                # Policy loss (weighted by branch factors)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_loss = (pg_loss_per_sample * w).sum() / w_sum

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
                    v_loss = 0.5 * (v_loss_max * w).sum() / w_sum
                else:
                    v_loss_per_sample = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (v_loss_per_sample * w).sum() / w_sum

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

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        # Save complete checkpoint (model weights + normalization running stats)
        checkpoint = {"model_state_dict": agent.state_dict()}
        env0 = envs[0]
        if env0._normalize_obs_wrapper is not None and hasattr(env0._normalize_obs_wrapper, "obs_rms"):
            obs_rms = env0._normalize_obs_wrapper.obs_rms
            checkpoint["obs_rms_mean"] = np.array(obs_rms.mean, copy=True)
            checkpoint["obs_rms_var"] = np.array(obs_rms.var, copy=True)
            checkpoint["obs_rms_count"] = float(obs_rms.count)
        if env0._normalize_reward_wrapper is not None and hasattr(env0._normalize_reward_wrapper, "return_rms"):
            ret_rms = env0._normalize_reward_wrapper.return_rms
            checkpoint["ret_rms_mean"] = np.array(ret_rms.mean, copy=True)
            checkpoint["ret_rms_var"] = np.array(ret_rms.var, copy=True)
            checkpoint["ret_rms_count"] = float(ret_rms.count)
        ckpt_dir = f"checkpoints/{args.env_id}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/seed{args.seed}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"complete checkpoint saved to {ckpt_path}")

        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    for env in envs:
        env.close()
    writer.close()
