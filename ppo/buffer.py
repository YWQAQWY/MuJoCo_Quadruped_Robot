"""RolloutBuffer：存储一条 rollout 的轨迹数据。

===============================  你的任务 1  ===============================
实现 compute_gae()：用广义优势估计 (GAE, Schulman et al. 2015) 计算
advantage 和 value target (returns)。

需要用到（都已存在 self 里，形状均为 [rollout_len]）：
    self.rewards   每步奖励 r_t
    self.values    每步状态价值 V(s_t)
    self.dones     每步是否结束 d_t（终止或截断都是 1）
    self.gamma     折扣因子 γ
    self.lam       GAE 参数 λ

提示（按这个顺序推）：
    1) 时序差分误差:  δ_t = r_t + γ·V(s_{t+1})·(1 − d_t) − V(s_t)
       V(s_{t+1}) 用 values[t+1] 近似；最后一步 (t = L−1) 用传入的
       last_value 引导（bootstrap）。d_t=1 时该项为 0。
    2) 从后往前递推:  A_t = δ_t + γλ·(1 − d_t)·A_{t+1}
       （最后一步的 A_{L−1} = δ_{L−1}）
    3) value target:  returns_t = A_t + V(s_t)
    4) 建议对 advantages 做标准化: A ← (A − mean(A)) / (std(A) + 1e-8)，
       这是 PPO 的常见做法，能显著提高训练稳定性。

写完后可以参考 ppo/reference_solution.py 对照。
=============================================================================
"""
from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(
        self,
        rollout_len: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.gamma = gamma
        self.lam = lam
        self.obs = np.zeros((rollout_len, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_len, act_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_len, dtype=np.float32)
        self.values = np.zeros(rollout_len, dtype=np.float32)
        self.rewards = np.zeros(rollout_len, dtype=np.float32)
        self.dones = np.zeros(rollout_len, dtype=np.float32)
        self.advantages = np.zeros(rollout_len, dtype=np.float32)
        self.returns = np.zeros(rollout_len, dtype=np.float32)
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        """按时间顺序存一条转移 (s_t, a_t, logπ, V(s_t), r_t, d_t)。"""
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float):
        """填好 self.advantages 和 self.returns。last_value = V(s_L)，即
        rollout 结束后当前状态的价值（用于最后一步 bootstrap）。

        ============================  TODO 1  ============================
        在这里实现 GAE。伪代码：
            next_value = last_value
            gae = 0
            for t in reversed(range(rollout_len)):
                delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + gamma * lam * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = gae + values[t]
                next_value = values[t]
            然后标准化 advantages
        ==================================================================
        """
        raise NotImplementedError("TODO 1: 请实现 compute_gae()（见文件头部说明）")

    def get(self):
        """返回训练所需的全部数据（numpy 数组）。"""
        assert self.ptr == len(self.obs), f"buffer 未满: {self.ptr}/{len(self.obs)}"
        return (
            self.obs,
            self.actions,
            self.log_probs,
            self.values,
            self.advantages,
            self.returns,
        )

    def clear(self):
        self.ptr = 0
