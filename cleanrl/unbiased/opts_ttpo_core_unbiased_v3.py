# Unbiased OPTS-TTPO V3 core: TreeGAE, branch weights, and prefix-value search.
# Search evaluates every tree node from its discounted prefix reward and value.
from typing import List

import torch


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
            weights[step, i] = root_branch_counts[env_idx][tree_root_id]

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
    rewards: torch.Tensor,
    search_advantages: torch.Tensor,
    values: torch.Tensor,
    parent_indices: torch.Tensor,
    state_branches: torch.Tensor,
    tree_indices: torch.Tensor,
    search_count: list[dict],
    max_search: int,
    root_branch_counts: List[dict],
    terminal_estimates: list[dict],
    skip_init_search: list[bool],
    affected_tree_ids: list[int],
    gamma: float = 0.99,
) -> list[int]:
    """Select one prefix-value degradation branch point for each terminated env."""
    n_steps = current_step + 1

    # Update every newly completed tree before constructing the shared baseline.
    # A tree being expanded keeps its previous cached estimate until this point.
    for env_idx, tree_id in zip(terminated_envs, affected_tree_ids):
        if skip_init_search[env_idx] and tree_id == -1:
            terminal_estimates[env_idx].pop(tree_id, None)
            continue
        parents = parent_indices[:n_steps, env_idx]
        trees = tree_indices[:n_steps, env_idx]
        root_steps = ((trees == tree_id) & (parents < 0)).nonzero(as_tuple=True)[0]
        root_value = values[root_steps, env_idx].mean()
        root_advantage = search_advantages[root_steps, env_idx].mean()
        terminal_estimates[env_idx][tree_id] = float((root_value + root_advantage).item())

    global_count = sum(len(estimates) for estimates in terminal_estimates)
    global_sum = sum(sum(estimates.values()) for estimates in terminal_estimates)
    env_num_trees = {}
    registered_envs = []
    registered_tree_ids = []
    registered_baselines = []

    # Registration only records searchable complete trees and their leave-one-out baselines.
    for env_idx in terminated_envs:
        all_tree_ids = torch.unique(tree_indices[:n_steps, env_idx]).tolist()
        env_num_trees[env_idx] = len(all_tree_ids)
        if global_count < 2:
            continue
        for tree_id in all_tree_ids:
            if tree_id not in terminal_estimates[env_idx]:
                continue
            if search_count[env_idx].get(tree_id, 0) >= max_search:
                continue
            registered_envs.append(env_idx)
            registered_tree_ids.append(tree_id)
            registered_baselines.append(
                (global_sum - terminal_estimates[env_idx][tree_id]) / (global_count - 1)
            )

    selected_by_env = {}
    num_trees = len(registered_tree_ids)
    if num_trees:
        device = search_advantages.device
        dtype = search_advantages.dtype
        num_terminated = len(terminated_envs)
        env_arr = torch.tensor(terminated_envs, device=device, dtype=torch.long)
        env_to_local = {env_idx: e_local for e_local, env_idx in enumerate(terminated_envs)}
        sub_parents = parent_indices[:n_steps, env_arr]
        sub_trees = tree_indices[:n_steps, env_arr]
        sub_rewards = rewards[:n_steps, env_arr]
        sub_values = values[:n_steps, env_arr]

        tree_rows = torch.full(
            (n_steps, num_terminated), -1, device=device, dtype=torch.long
        )
        for row, (env_idx, tree_id) in enumerate(zip(registered_envs, registered_tree_ids)):
            e_local = env_to_local[env_idx]
            tree_rows[:, e_local].masked_fill_(sub_trees[:, e_local] == tree_id, row)

        # Every row represents the state before that row's action. For a child
        # row x, its fixed prefix includes the reward of every ancestor action.
        node_depths = torch.zeros_like(sub_parents)
        prefix_returns = torch.zeros_like(sub_values)
        local_envs = torch.arange(num_terminated, device=device)
        gamma_t = torch.as_tensor(gamma, device=device, dtype=dtype)
        for step in range(n_steps):
            parents_at_step = sub_parents[step]
            non_root = parents_at_step >= 0
            safe_parents = parents_at_step.clamp_min(0)
            parent_depths = node_depths[safe_parents, local_envs]
            parent_prefixes = prefix_returns[safe_parents, local_envs]
            parent_rewards = sub_rewards[safe_parents, local_envs]
            node_depths[step] = torch.where(non_root, parent_depths + 1, 0)
            prefix_returns[step] = torch.where(
                non_root,
                parent_prefixes
                + torch.pow(gamma_t, parent_depths.to(dtype)) * parent_rewards,
                0.0,
            )

        node_m = prefix_returns + torch.pow(
            gamma_t, node_depths.to(dtype)
        ) * sub_values
        flat_rows = tree_rows.reshape(-1)
        baselines = torch.tensor(registered_baselines, device=device, dtype=dtype)
        flat_m = node_m.reshape(-1)
        flat_depths = node_depths.reshape(-1)
        valid_nodes = flat_rows >= 0
        safe_rows = flat_rows.clamp_min(0)
        candidate_mask = valid_nodes & (flat_m < baselines[safe_rows])
        candidate_flat = candidate_mask.nonzero(as_tuple=True)[0]
        candidate_rows = flat_rows[candidate_flat]
        candidate_depths = flat_depths[candidate_flat]
        candidate_m = flat_m[candidate_flat]

        min_depths = torch.full(
            (num_trees,), n_steps + 1, device=device, dtype=torch.long
        )
        min_depths.scatter_reduce_(
            0, candidate_rows, candidate_depths, reduce="amin", include_self=True
        )
        at_min_depth = candidate_depths == min_depths[candidate_rows]
        depth_rows = candidate_rows[at_min_depth]
        depth_flat = candidate_flat[at_min_depth]
        depth_m = candidate_m[at_min_depth]

        tree_candidate_m = torch.full(
            (num_trees,), float("-inf"), device=device, dtype=dtype
        )
        tree_candidate_m.scatter_reduce_(
            0, depth_rows, depth_m, reduce="amax", include_self=True
        )
        max_at_depth = depth_m == tree_candidate_m[depth_rows]
        tied_rows = depth_rows[max_at_depth]
        tied_flat = depth_flat[max_at_depth]

        priorities = torch.rand(tied_flat.numel(), device=device, dtype=dtype)
        priority_max = torch.full((num_trees,), -1.0, device=device, dtype=dtype)
        priority_max.scatter_reduce_(
            0, tied_rows, priorities, reduce="amax", include_self=True
        )
        chosen = priorities == priority_max[tied_rows]
        tree_positions = torch.full(
            (num_trees,), -1, device=device, dtype=torch.long
        )
        tree_positions.scatter_reduce_(
            0,
            tied_rows[chosen],
            tied_flat[chosen],
            reduce="amin",
            include_self=False,
        )
        found = tree_positions >= 0

        registered_envs_t = torch.tensor(registered_envs, device=device, dtype=torch.long)
        registered_tree_ids_t = torch.tensor(
            registered_tree_ids, device=device, dtype=torch.long
        )
        for env_idx in terminated_envs:
            env_rows = ((registered_envs_t == env_idx) & found).nonzero(as_tuple=True)[0]
            if env_rows.numel() == 0:
                continue
            env_m = tree_candidate_m[env_rows]
            max_m = env_m.max()
            tied_rows = env_rows[env_m == max_m]
            chosen_row = tied_rows[
                torch.randint(tied_rows.numel(), (1,), device=device)
            ].item()
            tree_id = int(registered_tree_ids_t[chosen_row].item())
            search_count[env_idx][tree_id] = search_count[env_idx].get(tree_id, 0) + 1
            selected_by_env[env_idx] = int(
                (tree_positions[chosen_row] // num_terminated).item()
            )
            print(
                f"    Tree Search: env_idx={env_idx}, tree_id={tree_id}, "
                f"M={tree_candidate_m[chosen_row].item():.4f}, "
                f"search_count={search_count[env_idx][tree_id]}, "
                f"depth={min_depths[chosen_row].item()}"
            )

    selected = []
    for env_idx in terminated_envs:
        if env_idx in selected_by_env:
            selected.append(selected_by_env[env_idx])
        else:
            print("    New Tree: no eligible value-degradation state")
            selected.append(-(env_num_trees[env_idx] + 1))
    return selected
