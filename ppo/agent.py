"""PPO 类（核心留空）+ 观测归一化 + 模型存取。

===============================  你的任务 2  ===============================
实现 PPO.update()：对 buffer 里的一条 rollout 做多轮小批量 SGD 更新。
其余部分（网络、观测归一化、存取）都已完整实现。

公式（PPO, Schulman et al. 2017）：

    ratio_t = exp(logπ_θ(a_t|s_t) − logπ_θold(a_t|s_t))
    L_CLIP  = mean( min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) )
    L_VF    = mean( (V(s_t) − returns_t)² )
    S       = mean( entropy )
    loss    = −L_CLIP + value_coef·0.5·L_VF − entropy_coef·S

流程提示：
    1) 从 buffer.get() 取数据并转成 torch tensor（注意 dtype 和 device）
    2) 观测用 self.obs_norm.normalize() 归一化
    3) advantages 已由 buffer 标准化过，直接使用
    4) 对每个 epoch：把数据随机打乱、切成 batch_size 的小批量
    5) 每个小批量：self.network.evaluate() 得到新 log_prob / value / entropy，
       用 self.log_probs_old（旧策略概率）算 ratio 和 clipped loss，反向传播，
       max_grad_norm 梯度裁剪，Adam 走一步
    6) 返回统计字典（policy_loss / value_loss / entropy / explained_var）

写完后可以参考 ppo/reference_solution.py 对照。
=============================================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ppo.network import ActorCritic


class RunningMeanStd:
    """观测的滑动均值/方差归一化（完整提供）。

    思路：训练时不断用新 batch 更新均值方差，策略看到的始终是
    「相对历史分布」的标准化观测，让 PPO 对量纲不敏感。
    """

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        """x: [N, dim]。用并行更新的方式合并 batch 统计量。"""
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class PPO:
    def __init__(
        self,
        obs_dim: int = 45,
        act_dim: int = 12,
        lr: float = 1e-3,
        clip: float = 0.2,
        epochs: int = 5,
        batch_size: int = 256,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr

        self.network = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.obs_norm = RunningMeanStd((obs_dim,))

    # ------------------------------------------------------------------ #
    # 训练接口（完整提供）
    # ------------------------------------------------------------------ #
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """单步决策：返回 (action, log_prob, value)。obs 先做归一化。"""
        obs_norm = self.obs_norm.normalize(obs)
        return self.network.get_action(obs_norm, deterministic)

    def set_lr(self, lr: float):
        self.lr = lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    # ------------------------------------------------------------------ #
    # 核心更新（留空）
    # ------------------------------------------------------------------ #
    def update(self, buffer) -> dict[str, float]:
        """用 buffer 里的 rollout 做 epochs 轮小批量更新。

        ============================  TODO 2  ============================
        在这里实现 PPO 更新（见文件头部说明）。
        ==================================================================
        """
        raise NotImplementedError("TODO 2: 请实现 PPO.update()（见文件头部说明）")

    # ------------------------------------------------------------------ #
    # 存取（完整提供）
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, extra: dict | None = None):
        ckpt = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_mean": self.obs_norm.mean,
            "obs_var": self.obs_norm.var,
            "obs_count": self.obs_norm.count,
            "lr": self.lr,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, str(path))

    def load(self, path: str | Path, load_optimizer: bool = True):
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.obs_norm.mean = ckpt["obs_mean"]
        self.obs_norm.var = ckpt["obs_var"]
        self.obs_norm.count = ckpt["obs_count"]
        self.lr = ckpt.get("lr", self.lr)
        return ckpt

    def eval_mode(self):
        self.network.eval()
