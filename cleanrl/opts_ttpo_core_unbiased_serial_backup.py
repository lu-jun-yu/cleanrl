# Unbiased OPTS-TTPO core: TreeGAE, branch weights, and value-degradation search.
# Shared by the Atari and continuous-action unbiased training entrypoints.
import random
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
    advantages: torch.Tensor,
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
    gae_lambda: float = 0.95,
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
        root_advantage = advantages[root_steps, env_idx].mean()
        terminal_estimates[env_idx][tree_id] = float(
            (root_value + gae_lambda * root_advantage).item()
        )

    global_count = sum(len(estimates) for estimates in terminal_estimates)
    global_sum = sum(sum(estimates.values()) for estimates in terminal_estimates)
    selected = []

    for env_idx in terminated_envs:
        all_tree_ids = list(set(tree_indices[:n_steps, env_idx].tolist()))
        next_tree_id = -(len(all_tree_ids) + 1)
        candidates = []

        if global_count >= 2:
            for tree_id in all_tree_ids:
                if tree_id not in terminal_estimates[env_idx]:
                    continue
                if search_count[env_idx].get(tree_id, 0) >= max_search:
                    continue

                baseline = (
                    global_sum - terminal_estimates[env_idx][tree_id]
                ) / (global_count - 1)
                parents = parent_indices[:n_steps, env_idx]
                trees = tree_indices[:n_steps, env_idx]
                tree_steps = (trees == tree_id).nonzero(as_tuple=True)[0].tolist()
                root_edges = [step for step in tree_steps if int(parents[step].item()) < 0]

                children = {}
                for step in tree_steps:
                    parent = int(parents[step].item())
                    if parent >= 0:
                        children.setdefault(parent, []).append(step)

                root_steps_t = torch.tensor(root_edges, device=advantages.device, dtype=torch.long)
                root_value = float(values[root_steps_t, env_idx].mean().item())
                root_advantage = float(advantages[root_steps_t, env_idx].mean().item())
                current_m = root_value
                current_depth = 0
                current_position = random.choice(root_edges)
                current_outgoing = root_edges
                state = None

                while True:
                    if current_m < baseline:
                        state = {
                            "m": current_m,
                            "step": current_position,
                            "depth": current_depth,
                        }
                        break

                    child_depth = current_depth + 1
                    discount = (gamma * gae_lambda) ** child_depth
                    child_states = []
                    for incoming_edge in current_outgoing:
                        outgoing = children.get(incoming_edge)
                        if not outgoing:
                            continue
                        outgoing_t = torch.tensor(outgoing, device=advantages.device, dtype=torch.long)
                        node_advantage = float(advantages[outgoing_t, env_idx].mean().item())
                        node_path_weight = 1.0 / float(branch_weights[incoming_edge, env_idx].item())
                        child_m = root_value + gae_lambda * (
                            root_advantage - discount * node_path_weight * node_advantage
                        )
                        child_states.append((child_m, outgoing[0], incoming_edge, outgoing))

                    if not child_states:
                        break

                    max_child_m = max(item[0] for item in child_states)
                    current_m, current_position, _, current_outgoing = random.choice(
                        [item for item in child_states if item[0] == max_child_m]
                    )
                    current_depth = child_depth

                if state is not None:
                    candidates.append((state, tree_id))

        if not candidates:
            print("    New Tree: no eligible value-degradation state")
            selected.append(next_tree_id)
            continue

        max_candidate_m = max(item[0]["m"] for item in candidates)
        best_state, best_tree_id = random.choice(
            [item for item in candidates if item[0]["m"] == max_candidate_m]
        )
        search_count[env_idx][best_tree_id] = search_count[env_idx].get(best_tree_id, 0) + 1
        print(
            f"    Tree Search: env_idx={env_idx}, tree_id={best_tree_id}, "
            f"M={best_state['m']:.4f}, "
            f"search_count={search_count[env_idx][best_tree_id]}, "
            f"depth={best_state['depth']}"
        )
        selected.append(best_state["step"])

    return selected
