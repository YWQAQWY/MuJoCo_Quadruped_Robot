"""PPO 数学与更新流程的快速回归测试。"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PPO import PPO, compute_gae  # noqa: E402


def test_gae_boundaries_and_truncation():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = torch.zeros(4)
    next_values = torch.tensor([0.0, 10.0, 0.0, 10.0])
    terminated = torch.tensor([0.0, 1.0, 0.0, 0.0])
    episode_ends = torch.tensor([0.0, 1.0, 0.0, 1.0])
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, episode_ends, gamma=0.9, lmbda=1.0
    )
    # t=1 是真正终止，不 bootstrap；t=3 是时间截断，应 bootstrap。
    assert torch.allclose(advantages, torch.tensor([1.9, 1.0, 10.0, 10.0]))
    assert advantages.shape == (4,)
    assert torch.allclose(returns, advantages)


def test_bounded_actions_and_update():
    agent = PPO(5, 32, 2, 3e-4, 1e-3, 0.95, 2, 0.2, 8, 0.99, 0.99, "cpu")
    states = np.random.randn(32, 5).astype(np.float32)
    actions = np.asarray([agent.take_action(s) for s in states])
    assert np.all(actions < 1.0) and np.all(actions > -1.0)
    transitions = {
        "states": states,
        "actions": actions,
        "next_states": np.random.randn(32, 5).astype(np.float32),
        "rewards": np.random.randn(32).astype(np.float32),
        "terminated": np.zeros(32, dtype=np.float32),
        "episode_ends": np.zeros(32, dtype=np.float32),
    }
    metrics = agent.update(transitions)
    assert all(np.isfinite(value) for value in metrics.values())
    assert {"approx_kl", "clip_fraction", "explained_var", "action_std"} <= metrics.keys()


if __name__ == "__main__":
    test_gae_boundaries_and_truncation()
    test_bounded_actions_and_update()
    print("[OK] PPO GAE、动作边界和更新测试通过")
