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
import copy
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load, load_all  # noqa: E402
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
    p.add_argument("--resume", type=str, default=None, help="从完整 checkpoint 恢复训练")
    return p.parse_args()


def make_checkpoint(agent, iteration, total_steps, best_eval_reward, config):
    return {
        "actor": agent.actor_net.state_dict(),
        "critic": agent.critic_net.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "iteration": iteration,
        "total_steps": total_steps,
        "best_eval_reward": best_eval_reward,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def evaluate(agent, seed, eval_cfg):
    """用固定指令和无噪声环境评估，返回完整回合平均奖励/长度。"""
    commands = eval_cfg["commands"]
    episode_length = eval_cfg["episode_length"]
    env = QuadrupedEnv(seed=seed, episode_length=episode_length,
                       add_noise=eval_cfg["add_noise"], randomize=eval_cfg["randomize"],
                       command_override=eval_cfg["command_override"])
    rewards, lengths = [], []
    for ep in range(eval_cfg["episodes"]):
        obs, _ = env.reset()
        env.set_commands(*commands[ep % len(commands)])
        total = 0.0
        for step in range(episode_length):
            obs, reward, terminated, truncated, _ = env.step(
                agent.take_action(obs, deterministic=eval_cfg["deterministic"])
            )
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        lengths.append(step + 1)
    return float(np.mean(rewards)), float(np.mean(lengths))


def main():
    args = parse_args()
    ppo_cfg = TRAIN_CFG["ppo"]
    tr_cfg = TRAIN_CFG["training"]
    eval_cfg = TRAIN_CFG["evaluation"]
    log_cfg = TRAIN_CFG["logging"]

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
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    env_options = tr_cfg["environment"]
    env = QuadrupedEnv(seed=seed, add_noise=env_options["add_noise"],
                       randomize=env_options["randomize"],
                       command_override=env_options["command_override"])
    hidden_dims = ([args.hidden_dim, args.hidden_dim] if args.hidden_dim is not None
                   else ppo_cfg["hidden_dims"])
    actor_lr = args.actor_lr if args.actor_lr is not None else ppo_cfg["actor_lr"]
    critic_lr = args.critic_lr if args.critic_lr is not None else ppo_cfg["critic_lr"]
    gamma = args.gamma if args.gamma is not None else ppo_cfg["gamma"]
    agent = PPO(
        state_dim=env.obs_dim,
        hidden_dim=hidden_dims[0], hidden_dims=hidden_dims,
        action_dim=env.act_dim,
        actor_lr=actor_lr, critic_lr=critic_lr,
        lmbda=args.lmbda if args.lmbda is not None else ppo_cfg["lmbda"],
        epoch=args.epochs if args.epochs is not None else ppo_cfg["epochs"],
        eps=args.eps if args.eps is not None else ppo_cfg["eps"],
        batch_size=ppo_cfg["batch_size"],
        entropy_coef=ppo_cfg["entropy_coef"],
        value_coef=ppo_cfg["value_coef"],
        value_clip=ppo_cfg["value_clip"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        target_kl=ppo_cfg["target_kl"],
        actor_gamma=gamma, critic_gamma=gamma,
        activation=ppo_cfg["activation"], hidden_init_gain=ppo_cfg["hidden_init_gain"],
        actor_output_gain=ppo_cfg["actor_output_gain"],
        critic_output_gain=ppo_cfg["critic_output_gain"],
        initial_log_std=ppo_cfg["initial_log_std"],
        log_std_bounds=ppo_cfg["log_std_bounds"],
        numerical_epsilon=ppo_cfg["numerical_epsilon"],
        device=device,
    )

    effective_config = load_all()
    effective_train = copy.deepcopy(TRAIN_CFG)
    effective_train["ppo"].update({
        "hidden_dims": hidden_dims, "actor_lr": actor_lr, "critic_lr": critic_lr,
        "gamma": gamma, "lmbda": agent.lmbda, "epochs": agent.epoch, "eps": agent.eps,
    })
    effective_train["training"].update({
        "iterations": iterations, "rollout_len": rollout_len, "seed": seed,
        "save_every": save_every, "device": device,
    })
    effective_config["train"] = effective_train

    run_name = args.run_name or (Path(args.resume).parent.name if args.resume
                                 else time.strftime("run_%Y%m%d_%H%M%S"))
    run_dir = ROOT / "runs" / run_name
    log_dir = ROOT / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = 0
    total_steps = 0
    best_eval_reward = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        agent.actor_net.load_state_dict(checkpoint["actor"])
        agent.critic_net.load_state_dict(checkpoint["critic"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_iteration = int(checkpoint["iteration"]) + 1
        total_steps = int(checkpoint.get("total_steps", start_iteration * rollout_len))
        best_eval_reward = float(checkpoint.get("best_eval_reward", -float("inf")))
        if "rng" in checkpoint:
            random.setstate(checkpoint["rng"]["python"])
            np.random.set_state(checkpoint["rng"]["numpy"])
            torch.set_rng_state(checkpoint["rng"]["torch"].cpu())

    csv_path = log_dir / "progress.csv"
    append_log = args.resume is not None and csv_path.exists()
    csv_file = open(csv_path, "a" if append_log else "w", newline="")
    writer = csv.writer(csv_file)
    if not append_log:
        writer.writerow(["iteration", "total_steps", "mean_ep_reward", "mean_ep_len",
                         "policy_loss", "value_loss", "entropy", "approx_kl",
                         "clip_fraction", "explained_var", "action_std", "eval_reward",
                         "eval_ep_len", "actor_lr", "fps"])

    print(f"运行目录: runs/{run_name}   设备: {device}   每次迭代 {rollout_len} 步")

    obs, _ = env.reset()
    ep_reward, ep_len = 0.0, 0
    for it in range(start_iteration, iterations):
        # ---- 采集 rollout（格式与你在 CartPole 里用的一致） ----
        transition_dict = {
            "states": [], "actions": [], "next_states": [],
            "rewards": [], "terminated": [], "episode_ends": [],
        }
        ep_rewards, ep_lens = [], []
        t_start = time.time()

        for _ in range(rollout_len):
            action = agent.take_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition_dict["states"].append(obs)
            transition_dict["actions"].append(action)
            transition_dict["next_states"].append(next_obs)
            transition_dict["rewards"].append(reward)
            transition_dict["terminated"].append(terminated)
            transition_dict["episode_ends"].append(done)

            obs = next_obs
            ep_reward += reward
            ep_len += 1
            if done:
                ep_rewards.append(ep_reward)
                ep_lens.append(ep_len)
                ep_reward, ep_len = 0.0, 0
                obs, _ = env.reset()

        # ---- PPO 更新（你的代码） ----
        metrics = agent.update(transition_dict)
        total_steps += rollout_len

        if tr_cfg.get("lr_anneal", True):
            fraction = max(0.0, 1.0 - (it + 1) / iterations)
            agent.actor_optimizer.param_groups[0]["lr"] = actor_lr * fraction
            agent.critic_optimizer.param_groups[0]["lr"] = critic_lr * fraction

        # ---- 记录 ----
        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        mean_ep_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        fps = rollout_len / (time.time() - t_start)
        eval_reward = float("nan")
        eval_ep_len = float("nan")
        should_eval = (it + 1) % eval_cfg["every"] == 0 or it == iterations - 1
        if should_eval:
            eval_reward, eval_ep_len = evaluate(agent, seed + eval_cfg["seed_offset"], eval_cfg)
        writer.writerow([
            it, total_steps, f"{mean_ep_reward:.3f}", f"{mean_ep_len:.1f}",
            f"{metrics['policy_loss']:.6f}", f"{metrics['value_loss']:.6f}",
            f"{metrics['entropy']:.6f}", f"{metrics['approx_kl']:.6f}",
            f"{metrics['clip_fraction']:.6f}", f"{metrics['explained_var']:.6f}",
            f"{metrics['action_std']:.6f}", f"{eval_reward:.3f}", f"{eval_ep_len:.1f}",
            f"{agent.actor_optimizer.param_groups[0]['lr']:.8f}", f"{fps:.1f}",
        ])
        csv_file.flush()

        if it % log_cfg["print_every"] == 0 or it == iterations - 1:
            print(f"iter {it:5d} | 平均回合奖励 {mean_ep_reward:8.2f} | "
                  f"回合长度 {mean_ep_len:6.1f} | {fps:6.0f} fps")

        is_new_best = should_eval and eval_reward > best_eval_reward
        if is_new_best:
            best_eval_reward = eval_reward
        checkpoint = make_checkpoint(agent, it, total_steps, best_eval_reward, effective_config)
        if (it + 1) % save_every == 0:
            torch.save(checkpoint, run_dir / f"checkpoint_{it + 1}.pt")
        if tr_cfg["checkpoint_every_iteration"] or it == iterations - 1:
            torch.save(checkpoint, run_dir / "last.pt")
        if is_new_best:
            torch.save(checkpoint, run_dir / "best.pt")

    csv_file.close()
    print(f"训练完成。最终模型在 runs/{run_name}/last.pt，最佳评估模型在 best.pt")


if __name__ == "__main__":
    main()
