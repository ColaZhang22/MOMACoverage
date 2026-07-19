# benchmarl/sampler/shared_preference_state.py
import multiprocessing as mp
import numpy as np
from typing import List, Tuple

IntervalTuple = Tuple[float, float, float]  # low, high, prob

class SharedPreferenceState:
    """Adaptive Preference Sampler, update intervals after evaluation and sample u during reset."""
 
    def __init__(self, noi: int, anchor_points: List[List[float]], manager: mp.managers.SyncManager | None = None):
        # Initialize sampler state
        # STEP 1: 等分区间，每个区间概率为1.0/k，k为区间数
        # STEP 2: 加入anchor points n*n
        # STEP 3: 计数器 step int
       
        self.intervals = manager.list([(i / noi, (i + 1) / noi, 1.0 / noi) for i in range(noi)])
        self.anchor_points = manager.list(anchor_points)
        self.num = noi
        self.odim = len(anchor_points[0])
        self.version = manager.Value("i", 0)

    def minmax_normalize(self,
        x: np.ndarray,
        axis=0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Min-max normalize the input array"""
        x = np.asarray(x, dtype=np.float64)

        x_min = np.min(x, axis=axis, keepdims=True)
        x_max = np.max(x, axis=axis, keepdims=True)

        return (x - x_min) / (x_max - x_min + eps)


    def update(self, u: np.ndarray) -> None:
        
        # step1 根据两个参考点 +  u得到的点（n+2），排序
        u = np.asarray(u, dtype=np.float64)
        if u.shape[0] != self.num or u.shape[1] != self.odim:
            raise ValueError(f"u must be ({self.num}, {self.odim}), got {u.shape}")
        anchors = np.asarray(list(self.anchor_points), dtype=np.float64)  # (odim, odim)
        u_all_raw = np.vstack([anchors, u])  # (n+2, odim)
        sort_idx = np.argsort(u_all_raw[:, 0], kind="stable")
        u_all_sorted = u_all_raw[sort_idx]  # (n+2, odim)
        u_norm = self.minmax_normalize(u_all_sorted, axis=0, eps=1e-8)
   
        # step2 对于排序后的每个点，计算其与相邻点的距离 n+1个值
        neighbor_dist = np.linalg.norm(np.diff(u_norm, axis=0),axis=1,)  # (n+1,)
        # step3 对于n+1个值， 通过 (dt+dt+1)/2 得到 n个值， 作为新旧值之间的比例
        p_u_sorted = 0.5 * (neighbor_dist[:-1] + neighbor_dist[1:])  # (n,)
       
        # step4 对这n个值归一化
        eps = 1e-8
        p_u_sorted = np.maximum(p_u_sorted, eps)
        p_new = p_u_sorted / p_u_sorted.sum()
  
        # step5 计算 新旧值之间的比例 ratio = (p_u + eps) / (p_w + eps)
        p_old = np.asarray([itv[2] for itv in self.intervals], dtype=np.float64)
        # ratio = (p_new + eps) / (p_old + eps)

        # # 通过clip进行更新，保证更新后的概率和为1
        # clip_eps = 0.2
        # ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        # p_updated = p_old * ratio # (n,)
        # p_updated /= p_updated.sum()

        # step6 更新intervals
        old_intervals = list(self.intervals)
        for i, (low, high, _) in enumerate(old_intervals):
            # self.intervals[i] = (float(low), float(high), float(p_updated[i]))
            self.intervals[i] = (float(low), float(high), float(p_new[i]))
        # Step 7: update version
        self.version.value = self.version.value + 1
        # print("----------------------------sampler information--------------------------------")
        # print(f"noi: {self.num}")
        # print(f"normalized u: {u_norm}")
        # print(f"p_new: {p_new}")
        # print(f"p_u_sorted: {p_u_sorted}")
        # print(f"p_old: {p_old}")
        # print("----------------------------sampler information--------------------------------")
    def sample(self) -> float:
        """reset时调用, 更具当前intervals概率采样u"""
        intervals = list(self.intervals)   # 或 self.get_intervals()
        if not intervals:
            return float(np.random.uniform(0.0, 1.0))
        
        probs = np.array([p for _, _, p in intervals], dtype=np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(intervals)) / len(intervals)
        idx = int(np.random.choice(len(intervals), p=probs))
        low, high, _ = intervals[idx]
        return float(np.random.uniform(low, high))
  