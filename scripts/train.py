"""PPO 训练主循环（完整提供，无需填写）。

用法：
    python scripts/train.py                          # 用你自己的 PPO 实现（TODO 必须已填）
    python scripts/train.py --reference              # 用参考答案跑通管线
    python scripts/train.py --iterations 1500 --run-name my_run
    python scripts/train.py --resume runs/my_run/checkpoint_100.pt --run-name my_run

训练产物：
    runs/<run_name>/checkpoint_<iter>.pt   模型检查点（含网络、优化器、观测统计）
    runs/<run_name>/best.pt                历史最佳平均回合奖励的模型
    logs/<run_name>/progress.csv           训练日志（奖励、loss、熵等）
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


def parse_args():
    p = argparse.ArgumentParser(description="训练四足狗 PPO 策略")
    p.add_argument("--iterations", type=int, default=1500, help="PPO 更新迭代次数")
    p.add_argument("--rollout-len", type=int, default=2048, help="每次迭代采集的步数")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    p.add_argument("--lr-decay-iters", type=int, default=1500, help="学习率线性衰减到 0.1 倍所需的迭代数")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", type=str, default=None, help="默认 runs/run_<时间戳>")
    p.add_argument("--reference", action="store_true", help="使用 ppo/reference_solution.py 的参考实现")
    p.add_argument("--resume", type=str, default=None, help="从检查点继续训练（需配合同名 --run-name）")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.reference:
        from ppo.reference_solution import PPOReference, RolloutBufferReference
        AgentClass, BufferClass = PPOReference, RolloutBufferReference
    else:
        from ppo.agent import PPO
        from ppo.buffer import RolloutBuffer
        AgentClass, BufferClass = PPO, RolloutBuffer

    env = QuadrupedEnv(seed=args.seed)
    agent = AgentClass(obs_dim=env.obs_dim, act_dim=env.act_dim, lr=args.lr, device=args.device)
    buffer = BufferClass(args.rollout_len, env.obs_dim, env.act_dim)

    assert args.rollout_len % agent.batch_size == 0, "rollout_len 必须是 batch_size 的整数倍"

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / run_name
    log_dir = ROOT / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "progress.csv"

    start_iter = 0
    total_steps = 0
    best_mean_reward = -np.inf
    if args.resume:
        ckpt = agent.load(args.resume)
        start_iter = int(ckpt.get("iteration", 0)) + 1
        total_steps = int(ckpt.get("total_steps", 0))
        print(f"从 {args.resume} 恢复，从迭代 {start_iter} 继续")

    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if start_iter == 0:
        writer.writerow(["iteration", "total_steps", "mean_ep_reward", "mean_ep_len",
                         "policy_loss", "value_loss", "entropy", "explained_var",
                         "lr", "fps"])

    print(f"运行目录: runs/{run_name}   日志: logs/{run_name}/progress.csv")
    print(f"设备: {agent.device}   每次迭代 {args.rollout_len} 步")

    obs, _ = env.reset()
    t_start = time.time()
    for it in range(start_iter, args.iterations):
        # 学习率线性衰减到 0.1 倍
        agent.set_lr(args.lr * max(0.1, 1.0 - it / args.lr_decay_iters))

        ep_rewards, ep_lens = [], []
        ep_reward, ep_len = 0.0, 0

        # ---- 采集 rollout ----
        for _ in range(args.rollout_len):
            action, log_prob, value = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.store(obs, action, log_prob, value, reward, done)

            ep_reward += reward
            ep_len += 1
            if done:
                ep_rewards.append(ep_reward)
                ep_lens.append(ep_len)
                ep_reward, ep_len = 0.0, 0
                obs, _ = env.reset()

        total_steps += args.rollout_len

        # ---- GAE + 更新 ----
        _, _, last_value = agent.act(obs)
        buffer.compute_gae(last_value)
        stats = agent.update(buffer)
        buffer.clear()

        # ---- 记录 ----
        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        mean_ep_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        fps = args.rollout_len / (time.time() - t_start)
        t_start = time.time()

        writer.writerow([it, total_steps, f"{mean_ep_reward:.3f}", f"{mean_ep_len:.1f}",
                         f"{stats['policy_loss']:.4f}", f"{stats['value_loss']:.4f}",
                         f"{stats['entropy']:.4f}", f"{stats['explained_var']:.4f}",
                         f"{agent.lr:.2e}", f"{fps:.1f}"])
        csv_file.flush()

        if it % 10 == 0 or it == args.iterations - 1:
            print(f"iter {it:5d} | 平均回合奖励 {mean_ep_reward:8.2f} | "
                  f"回合长度 {mean_ep_len:6.1f} | ev {stats['explained_var']:5.2f} | "
                  f"{fps:6.0f} fps")

        # ---- 保存检查点 ----
        if (it + 1) % args.save_every == 0:
            agent.save(run_dir / f"checkpoint_{it + 1}.pt",
                       extra={"iteration": it, "total_steps": total_steps, "seed": args.seed})
        if mean_ep_reward > best_mean_reward:
            best_mean_reward = mean_ep_reward
            agent.save(run_dir / "best.pt",
                       extra={"iteration": it, "total_steps": total_steps, "seed": args.seed})

    csv_file.close()
    print(f"训练完成。最佳平均回合奖励 {best_mean_reward:.2f}，模型在 runs/{run_name}/best.pt")


if __name__ == "__main__":
    main()
