"""用你自己写的 PPO.py（连续动作版）训练四足狗。

超参数默认值在 config/train.yaml（PPO 算法超参 + 训练循环配置），
命令行参数可以临时覆盖，例如：
    .venv/bin/python scripts/train_ppo.py --iterations 300 --run-name quick_test

PPO.py 是连续动作版本（高斯策略 + 熵正则）。
接口要求：
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epoch, eps, batch_size, actor_gamma, critic_gamma, device)
    action = agent.take_action(obs)          # obs: [45] numpy -> action: [12] numpy
    agent.update(transition_dict)            # 见下方 rollout 采集格式
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

from config import load  # noqa: E402
from env.quadruped_env import QuadrupedEnv  # noqa: E402
from PPO import PPO  # noqa: E402

TRAIN_CFG = load("train")  # config/train.yaml


def parse_args():
    p = argparse.ArgumentParser(description="用你自己的 PPO.py 训练四足狗")
    p.add_argument("--iterations", type=int, default=None, help="默认 config/train.yaml")
    p.add_argument("--rollout-len", type=int, default=None, help="默认 config/train.yaml")
    p.add_argument("--hidden-dim", type=int, default=None, help="默认 config/train.yaml")
    p.add_argument("--actor-lr", type=float, default=None, help="默认 config/train.yaml")
    p.add_argument("--critic-lr", type=float, default=None, help="默认 config/train.yaml")
    p.add_argument("--epochs", type=int, default=None, help="默认 config/train.yaml")
    p.add_argument("--eps", type=float, default=None, help="默认 config/train.yaml")
    p.add_argument("--gamma", type=float, default=None, help="默认 config/train.yaml")
    p.add_argument("--lmbda", type=float, default=None, help="默认 config/train.yaml")
    p.add_argument("--seed", type=int, default=None, help="默认 config/train.yaml")
    p.add_argument("--run-name", type=str, default=None, help="默认 runs/run_<时间戳>")
    p.add_argument("--save-every", type=int, default=None, help="默认 config/train.yaml")
    p.add_argument("--device", type=str, default=None,
                   choices=["auto", "cuda", "cpu"], help="默认 config/train.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    ppo_cfg = TRAIN_CFG["ppo"]
    tr_cfg = TRAIN_CFG["training"]

    def cfg(cli_val, key):
        """命令行参数优先，缺省用 config/train.yaml 的 training 段。"""
        return cli_val if cli_val is not None else tr_cfg[key]

    iterations = cfg(args.iterations, "iterations")
    rollout_len = cfg(args.rollout_len, "rollout_len")
    seed = cfg(args.seed, "seed")
    save_every = cfg(args.save_every, "save_every")
    device = cfg(args.device, "device")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = QuadrupedEnv(seed=seed)
    agent = PPO(
        state_dim=env.obs_dim,
        hidden_dim=args.hidden_dim if args.hidden_dim is not None else ppo_cfg["hidden_dim"],
        action_dim=env.act_dim,
        actor_lr=args.actor_lr if args.actor_lr is not None else ppo_cfg["actor_lr"],
        critic_lr=args.critic_lr if args.critic_lr is not None else ppo_cfg["critic_lr"],
        lmbda=args.lmbda if args.lmbda is not None else ppo_cfg["lmbda"],
        epoch=args.epochs if args.epochs is not None else ppo_cfg["epochs"],
        eps=args.eps if args.eps is not None else ppo_cfg["eps"],
        batch_size=ppo_cfg["batch_size"],
        entropy_coef=ppo_cfg["entropy_coef"],
        actor_gamma=args.gamma if args.gamma is not None else ppo_cfg["gamma"],
        critic_gamma=args.gamma if args.gamma is not None else ppo_cfg["gamma"],
        device=device,
    )

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / run_name
    log_dir = ROOT / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open(log_dir / "progress.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["iteration", "total_steps", "mean_ep_reward", "mean_ep_len", "fps"])

    print(f"运行目录: runs/{run_name}   设备: {device}   每次迭代 {rollout_len} 步")

    obs, _ = env.reset()
    for it in range(iterations):
        # ---- 采集 rollout（格式与你在 CartPole 里用的一致） ----
        transition_dict = {
            "states": [], "actions": [], "next_states": [],
            "rewards": [], "dones": [],
        }
        ep_rewards, ep_lens = [], []
        ep_reward, ep_len = 0.0, 0
        t_start = time.time()

        for _ in range(rollout_len):
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
        fps = rollout_len / (time.time() - t_start)
        writer.writerow([it, (it + 1) * rollout_len, f"{mean_ep_reward:.3f}",
                         f"{mean_ep_len:.1f}", f"{fps:.1f}"])
        csv_file.flush()

        if it % 10 == 0 or it == iterations - 1:
            print(f"iter {it:5d} | 平均回合奖励 {mean_ep_reward:8.2f} | "
                  f"回合长度 {mean_ep_len:6.1f} | {fps:6.0f} fps")

        if (it + 1) % save_every == 0:
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
