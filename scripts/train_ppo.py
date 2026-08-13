"""用你自己写的 PPO.py（连续动作版）训练四足狗。

前提：PPO.py 已改成连续动作版本（高斯策略，见 README/聊天说明）。
接口要求（你现在的类签名不变）：
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epoch, eps, actor_gamma, critic_gamma, device)
    action = agent.take_action(obs)          # obs: [45] numpy -> action: [12] numpy
    agent.update(transition_dict)            # 见下方 rollout 采集格式

用法：
    .venv/bin/python scripts/train_ppo.py --iterations 1500 --run-name my_ppo
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.quadruped_env import QuadrupedEnv  # noqa: E402
from PPO import PPO  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="用你自己的 PPO.py 训练四足狗")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--rollout-len", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lmbda", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    env = QuadrupedEnv(seed=args.seed)
    agent = PPO(
        state_dim=env.obs_dim, hidden_dim=args.hidden_dim, action_dim=env.act_dim,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        lmbda=args.lmbda, epoch=args.epochs, eps=args.eps,
        actor_gamma=args.gamma, critic_gamma=args.gamma, device=device,
    )

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / run_name
    log_dir = ROOT / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open(log_dir / "progress.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["iteration", "total_steps", "mean_ep_reward", "mean_ep_len", "fps"])

    print(f"运行目录: runs/{run_name}   设备: {device}   每次迭代 {args.rollout_len} 步")

    obs, _ = env.reset()
    for it in range(args.iterations):
        # ---- 采集 rollout（格式与你在 CartPole 里用的一致） ----
        transition_dict = {
            "states": [], "actions": [], "next_states": [],
            "rewards": [], "dones": [],
        }
        ep_rewards, ep_lens = [], []
        ep_reward, ep_len = 0.0, 0
        t_start = time.time()

        for _ in range(args.rollout_len):
            action = agent.take_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition_dict["states"].append(obs)
            transition_dict["actions"].append(action)
            transition_dict["next_states"].append(next_obs)
            transition_dict["rewards"].append(reward)
            transition_dict["dones"].append(done)

            obs = next_obs
            ep_reward += reward
            ep_len += 1
            if done:
                ep_rewards.append(ep_reward)
                ep_lens.append(ep_len)
                ep_reward, ep_len = 0.0, 0
                obs, _ = env.reset()

        # ---- PPO 更新（你的代码） ----
        agent.update(transition_dict)

        # ---- 记录 ----
        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        mean_ep_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        fps = args.rollout_len / (time.time() - t_start)
        writer.writerow([it, (it + 1) * args.rollout_len, f"{mean_ep_reward:.3f}",
                         f"{mean_ep_len:.1f}", f"{fps:.1f}"])
        csv_file.flush()

        if it % 10 == 0 or it == args.iterations - 1:
            print(f"iter {it:5d} | 平均回合奖励 {mean_ep_reward:8.2f} | "
                  f"回合长度 {mean_ep_len:6.1f} | {fps:6.0f} fps")

        if (it + 1) % args.save_every == 0:
            torch.save({"actor": agent.actor_net.state_dict(),
                        "critic": agent.critic_net.state_dict()},
                       run_dir / f"checkpoint_{it + 1}.pt")
            torch.save({"actor": agent.actor_net.state_dict(),
                        "critic": agent.critic_net.state_dict()},
                       run_dir / "best.pt")

    csv_file.close()
    print(f"训练完成。模型在 runs/{run_name}/best.pt")


if __name__ == "__main__":
    main()
