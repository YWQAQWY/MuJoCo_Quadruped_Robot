"""参考答案：填好的 GAE 与 PPO update。

建议学习顺序：
    1) 先自己实现 buffer.py 里的 compute_gae() 和 agent.py 里的 update()
    2) 跑通后再回来对照本文件
    3) 用相同随机种子分别训练你的版本和参考版本，对比奖励曲线

也可以用 `python scripts/train.py --reference` 直接跑本文件，
验证整条训练管线（环境、buffer、训练循环）没有问题。
"""
from __future__ import annotations

import numpy as np
import torch

from ppo.agent import PPO
from ppo.buffer import RolloutBuffer


class RolloutBufferReference(RolloutBuffer):
    """TODO 1 的参考答案：GAE。"""

    def compute_gae(self, last_value: float):
        next_value = last_value
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * next_value * (1.0 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.lam * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
            next_value = self.values[t]

        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)


class PPOReference(PPO):
    """TODO 2 的参考答案：PPO clipped objective 的多轮小批量更新。"""

    def update(self, buffer) -> dict[str, float]:
        obs, actions, log_probs_old, values_old, advantages, returns = buffer.get()

        self.obs_norm.update(obs)
        obs_norm = self.obs_norm.normalize(obs)

        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        log_probs_old_t = torch.as_tensor(log_probs_old, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = obs_t.shape[0]
        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]
                if len(idx) < 2:
                    continue

                values, log_probs_new, entropy = self.network.evaluate(obs_t[idx], actions_t[idx])

                ratio = torch.exp(log_probs_new - log_probs_old_t[idx])
                adv = advantages_t[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - returns_t[idx]) ** 2).mean()

                loss = (policy_loss
                        + self.value_coef * 0.5 * value_loss
                        - self.entropy_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()

        num_updates = self.epochs * max(1, n // self.batch_size)
        for k in stats:
            stats[k] /= num_updates

        # explained variance: 价值函数对 returns 的解释程度（越接近 1 越好）
        with torch.no_grad():
            values_all, _, _ = self.network.evaluate(obs_t, actions_t)
            residual_var = (returns_t - values_all).var()
            total_var = returns_t.var()
        stats["explained_var"] = float(1.0 - residual_var / (total_var + 1e-8))
        return stats
