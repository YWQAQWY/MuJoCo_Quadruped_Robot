"""用你自己写的 PPO.py（连续动作版）训练四足狗。

超参数默认值在 config/train.yaml（PPO 算法超参 + 训练循环配置），
命令行参数可以临时覆盖，例如：
    .venv/bin/python scripts/train_ppo.py --iterations 300 --run-name quick_test

PPO.py 是连续动作版本（高斯策略 + 熵正则）。
接口要求：
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epoch, eps, batch_size, actor_gamma, critic_gamma, device)
    action = agent.take_action(obs)          # obs: [54] numpy -> action: [12] numpy
    agent.update(transition_dict)            # 见下方 rollout 采集格式
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import deep_merge, load, load_all  # noqa: E402
from env.quadruped_env import QuadrupedEnv  # noqa: E402
from PPO import PPO  # noqa: E402

TRAIN_CFG = load("train")  # config/train.yaml
STAGES_CFG = load("stages")


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
    p.add_argument("--init-checkpoint", type=str, default=None,
                   help="用上一阶段 checkpoint 初始化新阶段（iteration 从 0 开始）")
    p.add_argument("--stage", choices=sorted(STAGES_CFG), default=None,
                   help="使用 config/stages.yaml 中的阶段覆盖配置")
    return p.parse_args()


def make_checkpoint(agent, iteration, total_steps, best_eval_score, config,
                    curriculum_state=None, best_gate_state=None):
    return {
        "actor": agent.actor_net.state_dict(),
        "critic": agent.critic_net.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "iteration": iteration,
        "total_steps": total_steps,
        "best_eval_score": best_eval_score,
        "config": config,
        "curriculum_state": copy.deepcopy(curriculum_state),
        "best_gate_state": copy.deepcopy(best_gate_state),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def load_state_with_input_expansion(module, source_state):
    """加载网络；仅允许第一层输入维度扩展，旧列复制、新列置零。"""
    target_state = module.state_dict()
    migrated = False
    expanded_state = {}
    for key, target in target_state.items():
        source = source_state[key]
        if source.shape == target.shape:
            expanded_state[key] = source
            continue
        is_first_layer = key.endswith("network.0.weight") or key.endswith("fc1.weight")
        can_expand = (is_first_layer and source.ndim == 2 and target.ndim == 2 and
                      source.shape[0] == target.shape[0] and source.shape[1] < target.shape[1])
        if not can_expand:
            raise RuntimeError(f"checkpoint 参数不兼容: {key} {tuple(source.shape)} -> "
                               f"{tuple(target.shape)}")
        expanded = torch.zeros_like(target)
        expanded[:, :source.shape[1]] = source.to(expanded.device)
        expanded_state[key] = expanded
        migrated = True
        print(f"扩展输入层 {key}: {source.shape[1]} → {target.shape[1]}（新增权重置零）")
    module.load_state_dict(expanded_state)
    return migrated


def evaluate(agent, seed, eval_cfg, env_override=None, pd_override=None):
    """固定指令评估；显式衡量跟踪误差，避免把稳定站立选为 best。"""
    commands = eval_cfg["commands"]
    episode_length = eval_cfg["episode_length"]
    env = QuadrupedEnv(seed=seed, episode_length=episode_length,
                       add_noise=eval_cfg["add_noise"], randomize=eval_cfg["randomize"],
                       command_override=eval_cfg["command_override"],
                       config_override=env_override, pd_config_override=pd_override)
    rewards, lengths, lin_errors, yaw_errors, successes, command_metrics = [], [], [], [], [], []
    falls = 0
    for ep in range(eval_cfg["episodes"]):
        obs, _ = env.reset()
        env.set_commands(*commands[ep % len(commands)])
        total = 0.0
        episode_lin_errors, episode_yaw_errors, velocities, yaw_rates = [], [], [], []
        contact_samples, diagonal_scores, body_contacts = [], [], []
        for step in range(episode_length):
            obs, reward, terminated, truncated, info = env.step(
                agent.take_action(obs, deterministic=eval_cfg["deterministic"])
            )
            total += reward
            lin_error = float(np.linalg.norm(np.asarray(env.commands[:2]) - info["base_lin_vel"][:2]))
            yaw_error = float(abs(env.commands[2] - info["base_ang_vel"][2]))
            episode_lin_errors.append(lin_error)
            episode_yaw_errors.append(yaw_error)
            velocities.append(np.asarray(info["base_lin_vel"][:2]))
            yaw_rates.append(float(info["base_ang_vel"][2]))
            contacts = np.asarray(info["foot_contacts"], dtype=bool)
            contact_samples.append(contacts.astype(float))
            diagonal_scores.append(float((contacts[0] == contacts[3]) and
                                         (contacts[1] == contacts[2]) and
                                         (contacts[0] != contacts[1])))
            body_contacts.append(int(info["body_contact_count"]) +
                                 int(info["torso_contact_count"]))
            if terminated or truncated:
                break
        falls += int(terminated)
        rewards.append(total)
        lengths.append(step + 1)
        lin_errors.extend(episode_lin_errors)
        yaw_errors.extend(episode_yaw_errors)
        score_cfg = eval_cfg["best_score"]
        command = np.asarray(commands[ep % len(commands)], dtype=float)
        mean_velocity = np.mean(velocities, axis=0)
        mean_yaw_rate = float(np.mean(yaw_rates))
        linear_command_norm = float(np.linalg.norm(command[:2]))
        actual_speed = float(np.linalg.norm(mean_velocity))
        moving = linear_command_norm > score_cfg["command_zero_threshold"]
        direction_ok = (not moving or float(np.dot(mean_velocity, command[:2])) > 0.0)
        speed_ratio_ok = (not moving or actual_speed >=
                          score_cfg["minimum_speed_ratio"] * linear_command_norm)
        stationary_ok = (moving or actual_speed < score_cfg["stationary_speed_threshold"])
        yaw_moving = abs(command[2]) > score_cfg["command_zero_threshold"]
        yaw_direction_ok = not yaw_moving or mean_yaw_rate * command[2] > 0.0
        yaw_ratio_ok = not yaw_moving or abs(mean_yaw_rate) >= (
            score_cfg["minimum_yaw_rate_ratio"] * abs(command[2]))
        success = (np.mean(episode_lin_errors) < score_cfg["linear_error_success_threshold"] and
                   np.mean(episode_yaw_errors) < score_cfg["yaw_error_success_threshold"] and
                   direction_ok and speed_ratio_ok and stationary_ok and
                   yaw_direction_ok and yaw_ratio_ok and not terminated)
        successes.append(float(success))
        contacts_array = np.asarray(contact_samples)
        command_metrics.append({
            "command": command.tolist(), "actual_velocity": mean_velocity.tolist(),
            "actual_yaw_rate": mean_yaw_rate,
            "linear_error": float(np.mean(episode_lin_errors)),
            "yaw_error": float(np.mean(episode_yaw_errors)),
            "survived": not terminated, "success": bool(success),
            "foot_contact_rates": np.mean(contacts_array, axis=0).tolist(),
            "foot_air_rates": np.mean(~contacts_array.astype(bool), axis=0).tolist(),
            "diagonal_contact_rate": float(np.mean(diagonal_scores)),
            "body_contact_count": int(np.sum(body_contacts)),
        })
    survival_rate = 1.0 - falls / eval_cfg["episodes"]
    mean_lin_error = float(np.mean(lin_errors))
    mean_yaw_error = float(np.mean(yaw_errors))
    score_cfg = eval_cfg["best_score"]
    tracking_score = (-score_cfg["tracking_error_weight"] *
                      (mean_lin_error + mean_yaw_error)
                      - score_cfg["fall_penalty"] * (1.0 - survival_rate))
    return {
        "reward": float(np.mean(rewards)), "episode_length": float(np.mean(lengths)),
        "linear_error": mean_lin_error, "yaw_error": mean_yaw_error,
        "survival_rate": survival_rate, "success_rate": float(np.mean(successes)),
        "score": tracking_score, "commands": command_metrics,
    }


def curriculum_scales(iteration, iterations, cfg):
    """将训练进度映射为指令范围和域随机化强度。"""
    if not cfg["enabled"]:
        return 1.0, 1.0
    progress = iteration / max(1, iterations - 1)
    span = max(1e-8, cfg["end_fraction"] - cfg["start_fraction"])
    alpha = float(np.clip((progress - cfg["start_fraction"]) / span, 0.0, 1.0))
    command_scale = cfg["command_scale_start"] + alpha * (1.0 - cfg["command_scale_start"])
    randomization_scale = cfg["randomization_scale_start"] + alpha * (
        cfg.get("randomization_scale_end", 1.0) - cfg["randomization_scale_start"]
    )
    return command_scale, randomization_scale


def initial_curriculum_state(cfg):
    return {"level": 0, "consecutive_passes": 0} if cfg.get("mode") == "competence" else None


def curriculum_values(iteration, iterations, cfg, state):
    """Return command/randomization/residual/frequency scales for the active level."""
    if cfg.get("mode") != "competence":
        command, randomization = curriculum_scales(iteration, iterations, cfg)
        return command, randomization, None, None
    level = cfg["levels"][state["level"]]
    return (float(level["command_scale"]), float(level["randomization_scale"]),
            float(level["residual_scale"]), float(level["frequency_scale"]))


def update_competence_curriculum(state, cfg, eval_metrics):
    """Promote only after repeated eligible evaluations; never advance by wall-clock iteration."""
    if cfg.get("mode") != "competence" or state["level"] >= len(cfg["levels"]) - 1:
        return False
    gate = cfg["promotion"]
    passed = (eval_metrics["survival_rate"] >= gate["minimum_survival_rate"] and
              eval_metrics["success_rate"] >= gate["minimum_success_rate"])
    state["consecutive_passes"] = state["consecutive_passes"] + 1 if passed else 0
    if state["consecutive_passes"] < gate["consecutive_evaluations"]:
        return False
    state["level"] += 1
    state["consecutive_passes"] = 0
    return True


def verify_reference_gait_preflight(env_cfg, training_cfg):
    """Require the active reference parameters to appear as qualified in the scan report."""
    if not training_cfg.get("reference_gait_preflight", False):
        return
    scan_cfg = load("gait_scan")
    report_path = ROOT / scan_cfg["report_path"]
    if not report_path.exists():
        raise RuntimeError("缺少参考步态扫描报告；请先运行 scripts/validate_reference_gait.py")
    gait = env_cfg["gait"]
    reference = gait["reference"]
    active = {
        "phase_frequency": gait["phase_frequency"], "duty_factor": gait["duty_factor"],
        "hip_pitch_amplitude": reference["joint_amplitudes"][1],
        "knee_amplitude": reference["joint_amplitudes"][2],
        "swing_knee_lift": reference["swing_knee_lift"],
        "phase_bias": reference["phase_bias"],
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matches = [item for item in report if item.get("qualified") and
               all(np.isclose(item["parameters"][key], value) for key, value in active.items())]
    if not matches:
        raise RuntimeError(f"当前参考步态未通过物理硬门槛: {active}；请重新运行扫描并更新配置")
    print(f"参考步态预检通过: vx={matches[0]['forward_velocity']:.3f} m/s")


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume 与 --init-checkpoint 不能同时使用")
    if args.resume and args.stage is None:
        resume_meta = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_stage = resume_meta.get("config", {}).get("active_stage", {}).get("name")
        if saved_stage in STAGES_CFG:
            args.stage = saved_stage
            print(f"从 checkpoint 自动恢复阶段配置: {saved_stage}")
    stage_cfg = STAGES_CFG.get(args.stage, {})
    active_train_cfg = deep_merge(TRAIN_CFG, stage_cfg.get("train"))
    env_override = stage_cfg.get("env")
    pd_override = stage_cfg.get("pd")
    ppo_cfg = active_train_cfg["ppo"]
    tr_cfg = active_train_cfg["training"]
    eval_cfg = active_train_cfg["evaluation"]
    log_cfg = active_train_cfg["logging"]
    curriculum_cfg = active_train_cfg["curriculum"]
    merged_env_cfg = deep_merge(load("env"), env_override)
    verify_reference_gait_preflight(merged_env_cfg, tr_cfg)

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
                       command_override=env_options["command_override"],
                       config_override=env_override, pd_config_override=pd_override)
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
    effective_train = copy.deepcopy(active_train_cfg)
    effective_train["ppo"].update({
        "hidden_dims": hidden_dims, "actor_lr": actor_lr, "critic_lr": critic_lr,
        "gamma": gamma, "lmbda": agent.lmbda, "epochs": agent.epoch, "eps": agent.eps,
    })
    effective_train["training"].update({
        "iterations": iterations, "rollout_len": rollout_len, "seed": seed,
        "save_every": save_every, "device": device,
    })
    effective_config["train"] = effective_train
    effective_config["env"] = deep_merge(load("env"), env_override)
    effective_config["pd"] = deep_merge(load("pd"), pd_override)
    effective_config["active_stage"] = {
        "name": args.stage, "description": stage_cfg.get("description"),
        "parent_checkpoint": args.init_checkpoint,
    }

    default_name = f"{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}" if args.stage else \
        time.strftime("run_%Y%m%d_%H%M%S")
    run_name = args.run_name or (Path(args.resume).parent.name if args.resume else default_name)
    run_dir = ROOT / "runs" / run_name
    log_dir = ROOT / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = 0
    total_steps = 0
    best_eval_score = -float("inf")
    curriculum_state = initial_curriculum_state(curriculum_cfg)
    best_gate_state = {"consecutive_eligible": 0}
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        actor_migrated = load_state_with_input_expansion(agent.actor_net, checkpoint["actor"])
        critic_migrated = load_state_with_input_expansion(agent.critic_net, checkpoint["critic"])
        input_migrated = actor_migrated or critic_migrated
        if stage_cfg.get("inherit_optimizer", True) and not input_migrated:
            agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        elif input_migrated:
            print("输入维度发生迁移：保留网络权重，重新初始化优化器状态")
        # 新阶段重新使用本阶段初始学习率，不继承上一阶段末尾的退火值。
        agent.actor_optimizer.param_groups[0]["lr"] = actor_lr
        agent.critic_optimizer.param_groups[0]["lr"] = critic_lr
        print(f"阶段初始化: {args.init_checkpoint} → {args.stage or 'custom'}（从 iteration 0 开始）")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        agent.actor_net.load_state_dict(checkpoint["actor"])
        agent.critic_net.load_state_dict(checkpoint["critic"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_iteration = int(checkpoint["iteration"]) + 1
        total_steps = int(checkpoint.get("total_steps", start_iteration * rollout_len))
        best_eval_score = float(checkpoint.get(
            "best_eval_score", checkpoint.get("best_eval_reward", -float("inf"))))
        if checkpoint.get("curriculum_state") is not None:
            curriculum_state = checkpoint["curriculum_state"]
        best_gate_state = checkpoint.get("best_gate_state", best_gate_state)
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
                         "eval_ep_len", "eval_linear_error", "eval_yaw_error",
                         "eval_survival_rate", "eval_success_rate", "eval_score",
                         "actor_lr", "command_scale",
                         "randomization_scale", "curriculum_level", "eval_commands", "fps"])

    print(f"运行目录: runs/{run_name}   设备: {device}   每次迭代 {rollout_len} 步")

    command_scale, randomization_scale, residual_scale, frequency_scale = curriculum_values(
        start_iteration, iterations, curriculum_cfg, curriculum_state)
    env.set_curriculum(command_scale, randomization_scale, residual_scale, frequency_scale)
    obs, _ = env.reset()
    ep_reward, ep_len = 0.0, 0
    for it in range(start_iteration, iterations):
        command_scale, randomization_scale, residual_scale, frequency_scale = curriculum_values(
            it, iterations, curriculum_cfg, curriculum_state)
        env.set_curriculum(command_scale, randomization_scale, residual_scale, frequency_scale)
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
            agent.actor_optimizer.param_groups[0]["lr"] = (
                tr_cfg["min_actor_lr"] + fraction * (actor_lr - tr_cfg["min_actor_lr"])
            )
            agent.critic_optimizer.param_groups[0]["lr"] = (
                tr_cfg["min_critic_lr"] + fraction * (critic_lr - tr_cfg["min_critic_lr"])
            )

        # ---- 记录 ----
        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        mean_ep_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        fps = rollout_len / (time.time() - t_start)
        eval_reward = float("nan")
        eval_ep_len = float("nan")
        eval_metrics = {key: float("nan") for key in (
            "linear_error", "yaw_error", "survival_rate", "success_rate", "score")}
        should_eval = (it + 1) % eval_cfg["every"] == 0 or it == iterations - 1
        if should_eval:
            eval_metrics = evaluate(
                agent, seed + eval_cfg["seed_offset"], eval_cfg, env_override, pd_override
            )
            eval_reward = eval_metrics["reward"]
            eval_ep_len = eval_metrics["episode_length"]
            if update_competence_curriculum(curriculum_state, curriculum_cfg, eval_metrics):
                print(f"课程晋级到 level {curriculum_state['level']}：下一轮应用新难度")
        writer.writerow([
            it, total_steps, f"{mean_ep_reward:.3f}", f"{mean_ep_len:.1f}",
            f"{metrics['policy_loss']:.6f}", f"{metrics['value_loss']:.6f}",
            f"{metrics['entropy']:.6f}", f"{metrics['approx_kl']:.6f}",
            f"{metrics['clip_fraction']:.6f}", f"{metrics['explained_var']:.6f}",
            f"{metrics['action_std']:.6f}", f"{eval_reward:.3f}", f"{eval_ep_len:.1f}",
            f"{eval_metrics['linear_error']:.5f}", f"{eval_metrics['yaw_error']:.5f}",
            f"{eval_metrics['survival_rate']:.4f}", f"{eval_metrics['success_rate']:.4f}",
            f"{eval_metrics['score']:.4f}",
            f"{agent.actor_optimizer.param_groups[0]['lr']:.8f}",
            f"{command_scale:.4f}", f"{randomization_scale:.4f}",
            curriculum_state["level"] if curriculum_state is not None else -1,
            json.dumps(eval_metrics.get("commands", []), ensure_ascii=False), f"{fps:.1f}",
        ])
        csv_file.flush()

        if it % log_cfg["print_every"] == 0 or it == iterations - 1:
            print(f"iter {it:5d} | 平均回合奖励 {mean_ep_reward:8.2f} | "
                  f"回合长度 {mean_ep_len:6.1f} | {fps:6.0f} fps")

        eligible = (should_eval and eval_metrics["survival_rate"] >=
                    eval_cfg["best_score"]["minimum_survival_rate"] and
                    eval_metrics["success_rate"] >=
                    eval_cfg["best_score"].get("minimum_success_rate", 0.0))
        if should_eval:
            best_gate_state["consecutive_eligible"] = (
                best_gate_state["consecutive_eligible"] + 1 if eligible else 0)
        consecutive_ok = best_gate_state["consecutive_eligible"] >= eval_cfg["best_score"].get(
            "minimum_consecutive_evaluations", 1)
        is_new_best = eligible and consecutive_ok and eval_metrics["score"] > best_eval_score
        if is_new_best:
            best_eval_score = eval_metrics["score"]
        checkpoint = make_checkpoint(agent, it, total_steps, best_eval_score, effective_config,
                                     curriculum_state, best_gate_state)
        if (it + 1) % save_every == 0:
            torch.save(checkpoint, run_dir / f"checkpoint_{it + 1}.pt")
        if tr_cfg["checkpoint_every_iteration"] or it == iterations - 1:
            torch.save(checkpoint, run_dir / "last.pt")
        if is_new_best:
            torch.save(checkpoint, run_dir / "best.pt")

    csv_file.close()
    if (run_dir / "best.pt").exists():
        print(f"训练完成。最终模型在 runs/{run_name}/last.pt，合格最佳模型在 best.pt")
    else:
        print(f"训练完成。最终模型在 runs/{run_name}/last.pt；尚未达到本阶段门槛，未生成 best.pt")


if __name__ == "__main__":
    main()
