"""Actor-Critic 网络（完整提供，无需填写）。

结构：共享 MLP 主干（tanh 激活）→ 两个头：
    - 策略头：输出动作均值 mu（对角高斯策略），log_std 是可学习参数
    - 值函数头：输出状态价值 V(s)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(action: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """对角高斯分布的对数概率密度（逐维，不求和）。

    log p(a|s) = -0.5 * (log(2π) + 2·log σ + ((a-μ)/σ)²)
    """
    return -0.5 * (LOG_2PI + 2.0 * torch.log(std) + ((action - mu) / std) ** 2)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int = 45,
        act_dim: int = 12,
        hidden_dims: tuple[int, ...] = (256, 256),
        log_std_init: float = -1.0,
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, act_dim)
        self.value_head = nn.Linear(in_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (动作均值 mu [B, act_dim], 状态价值 value [B, 1])。"""
        h = self.backbone(obs)
        return self.mu_head(h), self.value_head(h)

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """单步采样（训练时用）：返回 (action[act_dim], log_prob, value)。"""
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, value = self.forward(obs_t)
            std = self.log_std.exp()
            action = mu if deterministic else torch.normal(mu, std)
            log_prob = gaussian_log_prob(action, mu, std).sum(dim=-1).item()
            return action.squeeze(0).cpu().numpy(), float(log_prob), float(value.item())

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """批量评估（更新时用）：返回 (values [B], log_probs [B], entropy [B])。"""
        mu, value = self.forward(obs)
        std = self.log_std.exp()
        log_probs = gaussian_log_prob(actions, mu, std).sum(dim=-1)
        entropy = (0.5 * (1.0 + LOG_2PI) + self.log_std).sum().expand_as(log_probs)
        return value.squeeze(-1), log_probs, entropy
