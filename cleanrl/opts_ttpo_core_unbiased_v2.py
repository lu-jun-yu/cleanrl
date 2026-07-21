# Unbiased OPTS-TTPO V2 core: TreeGAE, branch weights, and value-degradation search.
# V2 computes M from a separate search advantage estimator.
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
    search_lam: float = 1.0,
) -> list[int]:
    """Select one value-degradation branch point independently for each env."""
    n_steps = current_step + 1
    env_indices = list(range(parent_indices.shape[1]))
    branch_weights = compute_branch_weight(
        num_steps=n_steps,
        parent_indices=parent_indices,
        state_branches=state_branches,
        env_indices=env_indices,
        root_branch_counts=root_branch_counts,
    )

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
        terminal_estimates[env_idx][tree_id] = float(
            (root_value + search_lam * root_advantage).item()
        )

    global_count = sum(len(estimates) for estimates in terminal_estimates)
    global_sum = sum(sum(estimates.values()) for estimates in terminal_estimates)
    env_num_trees = {}
    registered_envs = []
    registered_tree_ids = []
    registered_baselines = []

    # Registration is cheap metadata work. The depth-wise traversal below is fully batched.
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
        sub_advantages = search_advantages[:n_steps, env_arr]
        sub_values = values[:n_steps, env_arr]
        sub_branch_weights = branch_weights[:n_steps, env_arr]

        step_grid = torch.arange(n_steps, device=device).unsqueeze(1).expand(n_steps, num_terminated)
        env_grid = torch.arange(num_terminated, device=device).unsqueeze(0).expand(n_steps, num_terminated)
        flat_grid = step_grid * num_terminated + env_grid
        num_nodes = n_steps * num_terminated

        tree_rows = torch.full(
            (n_steps, num_terminated), -1, device=device, dtype=torch.long
        )
        for row, (env_idx, tree_id) in enumerate(zip(registered_envs, registered_tree_ids)):
            e_local = env_to_local[env_idx]
            tree_rows[:, e_local].masked_fill_(sub_trees[:, e_local] == tree_id, row)

        flat_rows = tree_rows.reshape(-1)
        root_mask = (sub_parents < 0) & (tree_rows >= 0)
        root_flat = flat_grid[root_mask]
        root_rows = tree_rows[root_mask]

        root_counts = torch.zeros(num_trees, device=device, dtype=dtype)
        root_values = torch.zeros(num_trees, device=device, dtype=dtype)
        root_advantages = torch.zeros(num_trees, device=device, dtype=dtype)
        root_counts.scatter_add_(0, root_rows, torch.ones_like(root_rows, dtype=dtype))
        root_values.scatter_add_(0, root_rows, sub_values[root_mask])
        root_advantages.scatter_add_(0, root_rows, sub_advantages[root_mask])
        root_values /= root_counts
        root_advantages /= root_counts

        # Pick an arbitrary root edge for restoring the root state; exact ties are randomized.
        root_priority = torch.rand(root_flat.numel(), device=device, dtype=dtype)
        root_priority_max = torch.full((num_trees,), -1.0, device=device, dtype=dtype)
        root_priority_max.scatter_reduce_(
            0, root_rows, root_priority, reduce="amax", include_self=True
        )
        chosen_root = root_priority == root_priority_max[root_rows]
        root_positions = torch.full((num_trees,), -1, device=device, dtype=torch.long)
        root_positions.scatter_reduce_(
            0, root_rows[chosen_root], root_flat[chosen_root], reduce="amin", include_self=False
        )

        # Edge depth is computed once for every environment in parallel.
        edge_depths = torch.ones_like(sub_parents)
        local_envs = torch.arange(num_terminated, device=device)
        for step in range(n_steps):
            parents_at_step = sub_parents[step]
            non_root = parents_at_step >= 0
            safe_parents = parents_at_step.clamp_min(0)
            parent_depth = edge_depths[safe_parents, local_envs]
            edge_depths[step] = torch.where(non_root, parent_depth + 1, 1)

        # Aggregate outgoing edge advantages at every non-terminal successor state.
        is_child_edge = (sub_parents >= 0) & (tree_rows >= 0)
        child_flat = flat_grid[is_child_edge]
        parent_flat = (sub_parents * num_terminated + env_grid)[is_child_edge]
        flat_advantages = sub_advantages.reshape(-1)
        node_advantage_sums = torch.zeros(num_nodes, device=device, dtype=dtype)
        node_outdegrees = torch.zeros(num_nodes, device=device, dtype=dtype)
        node_advantage_sums.scatter_add_(0, parent_flat, flat_advantages[child_flat])
        node_outdegrees.scatter_add_(
            0, parent_flat, torch.ones_like(parent_flat, dtype=dtype)
        )
        has_outgoing = node_outdegrees > 0
        node_advantages = node_advantage_sums / node_outdegrees.clamp_min(1.0)

        # Any outgoing edge represents the same branch state; choose one randomly for restore.
        state_priority = torch.rand(child_flat.numel(), device=device, dtype=dtype)
        state_priority_max = torch.full((num_nodes,), -1.0, device=device, dtype=dtype)
        state_priority_max.scatter_reduce_(
            0, parent_flat, state_priority, reduce="amax", include_self=True
        )
        chosen_state_edge = state_priority == state_priority_max[parent_flat]
        state_positions = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        state_positions.scatter_reduce_(
            0,
            parent_flat[chosen_state_edge],
            child_flat[chosen_state_edge],
            reduce="amin",
            include_self=False,
        )

        # M for the state reached after each edge. Invalid/terminal states remain -inf.
        successor_m = torch.full((num_nodes,), float("-inf"), device=device, dtype=dtype)
        valid_successor = has_outgoing & (flat_rows >= 0)
        valid_flat = valid_successor.nonzero(as_tuple=True)[0]
        valid_rows = flat_rows[valid_flat]
        flat_depths = edge_depths.reshape(-1)
        flat_weights = sub_branch_weights.reshape(-1)
        discounts = (gamma * search_lam) ** flat_depths[valid_flat].to(dtype)
        successor_m[valid_flat] = root_values[valid_rows] + search_lam * (
            root_advantages[valid_rows]
            - discounts * node_advantages[valid_flat] / flat_weights[valid_flat]
        )

        # Select the maximum-M successor for every non-root state in one grouped reduction.
        candidate_child_mask = valid_successor.reshape(n_steps, num_terminated) & (
            sub_parents >= 0
        )
        candidate_child_flat = flat_grid[candidate_child_mask]
        candidate_parent_flat = (
            sub_parents * num_terminated + env_grid
        )[candidate_child_mask]
        candidate_child_m = successor_m[candidate_child_flat]
        parent_max_m = torch.full((num_nodes,), float("-inf"), device=device, dtype=dtype)
        parent_max_m.scatter_reduce_(
            0, candidate_parent_flat, candidate_child_m, reduce="amax", include_self=True
        )
        max_child = candidate_child_m == parent_max_m[candidate_parent_flat]
        child_priority = torch.rand(candidate_child_flat.numel(), device=device, dtype=dtype)
        child_priority = child_priority.masked_fill(~max_child, -1.0)
        child_priority_max = torch.full((num_nodes,), -1.0, device=device, dtype=dtype)
        child_priority_max.scatter_reduce_(
            0, candidate_parent_flat, child_priority, reduce="amax", include_self=True
        )
        chosen_child = max_child & (
            child_priority == child_priority_max[candidate_parent_flat]
        )
        best_child = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        best_child.scatter_reduce_(
            0,
            candidate_parent_flat[chosen_child],
            candidate_child_flat[chosen_child],
            reduce="amin",
            include_self=False,
        )

        # Root states use tree row as their grouping key instead of a parent flat ID.
        root_successor_mask = root_mask.reshape(-1) & valid_successor
        root_successor_flat = root_successor_mask.nonzero(as_tuple=True)[0]
        root_successor_rows = flat_rows[root_successor_flat]
        root_successor_m = successor_m[root_successor_flat]
        root_max_m = torch.full((num_trees,), float("-inf"), device=device, dtype=dtype)
        root_max_m.scatter_reduce_(
            0, root_successor_rows, root_successor_m, reduce="amax", include_self=True
        )
        max_root_child = root_successor_m == root_max_m[root_successor_rows]
        root_child_priority = torch.rand(root_successor_flat.numel(), device=device, dtype=dtype)
        root_child_priority = root_child_priority.masked_fill(~max_root_child, -1.0)
        root_child_priority_max = torch.full(
            (num_trees,), -1.0, device=device, dtype=dtype
        )
        root_child_priority_max.scatter_reduce_(
            0,
            root_successor_rows,
            root_child_priority,
            reduce="amax",
            include_self=True,
        )
        chosen_root_child = max_root_child & (
            root_child_priority == root_child_priority_max[root_successor_rows]
        )
        best_root_child = torch.full(
            (num_trees,), -1, device=device, dtype=torch.long
        )
        best_root_child.scatter_reduce_(
            0,
            root_successor_rows[chosen_root_child],
            root_successor_flat[chosen_root_child],
            reduce="amin",
            include_self=False,
        )

        # All trees now move from root to leaf together; no per-tree path loop remains.
        baselines = torch.tensor(registered_baselines, device=device, dtype=dtype)
        current_m = root_values.clone()
        current_positions = root_positions.clone()
        current_depths = torch.zeros(num_trees, device=device, dtype=torch.long)
        next_nodes = best_root_child.clone()
        active = torch.ones(num_trees, device=device, dtype=torch.bool)
        found = torch.zeros(num_trees, device=device, dtype=torch.bool)
        found_m = torch.zeros(num_trees, device=device, dtype=dtype)
        found_positions = torch.full((num_trees,), -1, device=device, dtype=torch.long)
        found_depths = torch.zeros(num_trees, device=device, dtype=torch.long)

        while bool(active.any()):
            degraded = active & (current_m < baselines)
            found[degraded] = True
            found_m[degraded] = current_m[degraded]
            found_positions[degraded] = current_positions[degraded]
            found_depths[degraded] = current_depths[degraded]
            active &= ~degraded

            can_advance = active & (next_nodes >= 0)
            active &= can_advance
            if not bool(active.any()):
                break

            rows = active.nonzero(as_tuple=True)[0]
            incoming = next_nodes[rows]
            current_m[rows] = successor_m[incoming]
            current_positions[rows] = state_positions[incoming]
            current_depths[rows] += 1
            next_nodes[rows] = best_child[incoming]

        registered_envs_t = torch.tensor(registered_envs, device=device, dtype=torch.long)
        registered_tree_ids_t = torch.tensor(
            registered_tree_ids, device=device, dtype=torch.long
        )
        for env_idx in terminated_envs:
            env_rows = ((registered_envs_t == env_idx) & found).nonzero(as_tuple=True)[0]
            if env_rows.numel() == 0:
                continue
            env_m = found_m[env_rows]
            max_m = env_m.max()
            tied_rows = env_rows[env_m == max_m]
            chosen_row = tied_rows[
                torch.randint(tied_rows.numel(), (1,), device=device)
            ].item()
            tree_id = int(registered_tree_ids_t[chosen_row].item())
            search_count[env_idx][tree_id] = search_count[env_idx].get(tree_id, 0) + 1
            selected_by_env[env_idx] = int(
                (found_positions[chosen_row] // num_terminated).item()
            )
            print(
                f"    Tree Search: env_idx={env_idx}, tree_id={tree_id}, "
                f"M={found_m[chosen_row].item():.4f}, "
                f"search_count={search_count[env_idx][tree_id]}, "
                f"depth={found_depths[chosen_row].item()}"
            )

    selected = []
    for env_idx in terminated_envs:
        if env_idx in selected_by_env:
            selected.append(selected_by_env[env_idx])
        else:
            print("    New Tree: no eligible value-degradation state")
            selected.append(-(env_num_trees[env_idx] + 1))
    return selected
