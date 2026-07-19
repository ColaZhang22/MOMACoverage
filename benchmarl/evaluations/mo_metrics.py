# benchmarl/evaluation/mo_metrics.py
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# 与 experiment.py 里 dict_metrics 一致
IDX_PATH_LENGTH = 2
IDX_EXPLORATION_RATIO = 5

def cardinality_nondominated(F: np.ndarray) -> int:
    if F.size == 0:
        return 0
    return int(pareto_filter_maximize(F).shape[0])


def sparsity_schott(F: np.ndarray) -> float:
    """越小表示前沿上点越均匀；点重复/堆叠时趋近 0。"""
    P = pareto_filter_maximize(F) if F.size else F
    n = P.shape[0]
    if n <= 1:
        return 0.0
    L = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i] += np.abs(P[i, 0] - P[j, 0]) + np.abs(P[i, 1] - P[j, 1])
    L_mean = L.mean()
    if L_mean <= 1e-12:
        return 0.0
    return float(np.abs(L - L_mean).sum() / (n * L_mean))

def extract_explore_path_points(
    rollouts: Sequence,
    *,
    path_idx: int = IDX_PATH_LENGTH,
    explore_idx: int = IDX_EXPLORATION_RATIO,
    last_valid_idx_fn=None,
) -> np.ndarray:
    """
    每个 rollout 一个点: [ExplorationRatio, -PathLength]
    shape: (n_episodes, 2)
    """
    points = []
    for rollout in rollouts:
        manual_metrics = rollout["next", "manual_metrics"]
        done = rollout["next", "done"].squeeze(-1)
        done_steps = done.nonzero(as_tuple=True)[0]
        if done_steps.numel() > 0:
            # SARWrapper emits final manual_metrics and done in the same
            # transition, so the done index itself is the last valid sample.
            last_valid_idx = int(done_steps[0].item())
        else:
            last_valid_idx = manual_metrics.shape[0] - 1

        m = manual_metrics[last_valid_idx].reshape(-1)
        explore = float(m[explore_idx].item())
        path_len = float(m[path_idx].item())
        points.append([explore, -path_len])
    return np.asarray(points, dtype=np.float64)


def pareto_filter_maximize(F: np.ndarray) -> np.ndarray:
    """保留非支配点（两维均越大越好）。"""
    if F.shape[0] <= 1:
        return F
    keep = np.ones(F.shape[0], dtype=bool)
    for i in range(F.shape[0]):
        if not keep[i]:
            continue
        for j in range(F.shape[0]):
            if i == j or not keep[j]:
                continue
            if np.all(F[j] >= F[i]) and np.any(F[j] > F[i]):
                keep[i] = False
                break
    return F[keep]


def hypervolume_2d_maximize(F: np.ndarray, ref: np.ndarray) -> float:
    """
    F: (K, 2), 越大越好
    ref: (2,), 严格劣于所有点
    """
    if F.size == 0:
        return 0.0
    if np.any(F[:, 0] <= ref[0]) or np.any(F[:, 1] <= ref[1]):
        raise ValueError(
            f"ref {ref} must be dominated by all points. "
            f"min F={F.min(axis=0)}, max F={F.max(axis=0)}"
        )
    P = pareto_filter_maximize(F)
    order = np.argsort(P[:, 0])
    P = P[order]
    hv = 0.0
    for i in range(P.shape[0]):
        x_prev = ref[0] if i == 0 else P[i - 1, 0]
        width = P[i, 0] - x_prev
        height = P[i, 1] - ref[1]
        hv += width * height
    return float(hv)


def compute_explore_path_hypervolume(
    rollouts: Sequence,
    ref: np.ndarray,
    *,
    filter_nondominated: bool = True,
) -> Tuple[float, np.ndarray]:
    F = extract_explore_path_points(rollouts)
    F_use = pareto_filter_maximize(F) if filter_nondominated else F
    hv = hypervolume_2d_maximize(F_use, np.asarray(ref, dtype=np.float64))
    return hv, F

def minmax_normalize(
    x: np.ndarray,
    anchors: np.ndarray,
    axis=0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Min-max normalize the input array"""
    x = np.asarray(x, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)

    x_min = np.min(anchors, axis=axis)   # nadir，如 [8.5, -700.0]
    x_max = np.max(anchors, axis=axis)   # ideal，如 [95.0, -4.5]

    return (x - x_min) / (x_max - x_min + eps)

def compute_explore_path_mo_metrics(
    rollouts: Sequence,
    minmax_points: List[List[float]], # 2*odim
) -> dict:
    F = extract_explore_path_points(rollouts)
    P = pareto_filter_maximize(F)
    
    # normalize P
    minmax_anchors = np.asarray(list(minmax_points), dtype=np.float64)  # (odim, odim)
    P_norm = minmax_normalize(P, minmax_anchors, axis=0, eps=1e-8) # N*2
    ref_norm = np.asarray([-0.05, -0.05], dtype=np.float64)
    return {
        "hv": hypervolume_2d_maximize(P_norm, ref_norm) if P_norm.size else 0.0,
        "cardinality": int(P.shape[0]),
        "sparsity": sparsity_schott(P_norm),   # 用全 F 先过滤，函数内部会再 filter；或传 P：sparsity_schott_on_front(P)
        "n_episodes": int(F.shape[0]),
        "points": F,
    }
