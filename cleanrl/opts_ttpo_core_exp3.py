# OPTS-TTPO exp3 core: leaf-weighted TreeGAE, branch weights, and parallel tree-search node selection.
from typing import List

import numpy as np
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
    At a branch, each child advantage is weighted by the number of leaves in
    that child's subtree.
    Synchronizes advantages for identical state-action pairs.

    Args:
        terminal_step: The step index of the terminal node
        env_idx: The environment index
        rewards, values, dones, parent_indices: Tree data tensors
        advantages: Tensor to store computed advantages (modified in-place)
        gamma, gae_lambda: GAE parameters
        next_value: Bootstrap value for non-terminal leaf nodes
    """
    prefix_parents = parent_indices[: terminal_step + 1, env_idx].tolist()
    parent_nodes = {parent for parent in prefix_parents if parent >= 0}
    leaf_counts = [0] * (terminal_step + 1)
    for node in range(terminal_step, -1, -1):
        if node not in parent_nodes:
            leaf_counts[node] = 1
        parent = prefix_parents[node]
        if parent >= 0:
            leaf_counts[parent] += leaf_counts[node]

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
            child_leaf_counts = child_advs.new_tensor([leaf_counts[child] for child in children])
            weighted_child_adv = (child_advs * child_leaf_counts).sum() / child_leaf_counts.sum()
            advantages[current, env_idx] = delta + gamma * gae_lambda * weighted_child_adv

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
    advantages: torch.Tensor,
    parent_indices: torch.Tensor,
    tree_indices: torch.Tensor,
    search_count: list[dict],
    max_search: int,
    max_otrc_scores: list[dict],
    skip_init_search: list[bool],
    tree_search_state: list[dict],
    affected_tree_ids: list[int],
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
    affected_by_env = {env_idx: {affected_tree_ids[i]} for i, env_idx in enumerate(terminated_envs)}

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

    refresh_tids, roots, tree_e_local = [], [], []
    env_num_trees = [0] * E
    env_tree_ids_by_local = []
    for e_local, env_idx in enumerate(terminated_envs):
        col_trees = sub_trees[:, e_local]
        col_parents = sub_par[:, e_local]
        col_advs = sub_advs[:, e_local]
        all_tree_ids = torch.unique(col_trees).tolist()
        searchable_tids = [
            tid for tid in all_tree_ids
            if not (skip_init_search[env_idx] and tid == -1)
        ]
        env_tree_ids_by_local.append(searchable_tids)
        env_num_trees[e_local] = len(all_tree_ids)
        for tid in searchable_tids:
            if tid not in affected_by_env[env_idx] and tid in tree_search_state[env_idx]:
                continue
            root_steps = ((col_trees == tid) & (col_parents < 0)).nonzero(as_tuple=True)[0]
            refresh_tids.append(tid)
            roots.append(root_steps[col_advs[root_steps].argmax()].item() * E + e_local)
            tree_e_local.append(e_local)

    T = len(refresh_tids)
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

        for i, tid in enumerate(refresh_tids):
            env_idx = terminated_envs[tree_e_local[i]]
            score = float(max_otrc_score[i].item())
            tree_search_state[env_idx][tid] = {
                "score": score,
                "step": int(best_step[i].item()),
                "max_pos": int(max_pos[i].item()),
                "n_t": int(n_t[i].item()),
            }
            max_otrc_scores[env_idx][tid] = score

    pool = [v for d in max_otrc_scores for v in d.values()]
    mean_threshold = float(np.mean(pool)) if len(pool) > 1 else float("inf")

    for e_local, env_idx in enumerate(terminated_envs):
        candidates = []
        for tid in env_tree_ids_by_local[e_local]:
            if search_count[env_idx].get(tid, 0) >= max_search:
                continue
            state = tree_search_state[env_idx][tid]
            if state["score"] <= mean_threshold:
                continue
            candidates.append((state["score"], tid, state))

        if not candidates:
            print(f"    New Tree: best_otrc_score={float('-inf'):.4f}")
            selected.append(-(env_num_trees[e_local] + 1))
            continue

        score, best_tid, best_state = max(candidates, key=lambda item: item[0])
        search_count[env_idx][best_tid] = search_count[env_idx].get(best_tid, 0) + 1
        print(
            f"    Tree Search: env_idx={env_idx}, tree_id={best_tid}, "
            f"otrc_score={score:.4f}, "
            f"search_count={search_count[env_idx][best_tid]}, "
            f"depth={best_state['max_pos']} / {best_state['n_t']}"
        )
        selected.append(best_state["step"])

    return selected
