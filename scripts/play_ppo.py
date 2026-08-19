"""用你自己的 PPO.py 训练出的策略回放。

两种模式：
    python scripts/play_ppo.py --checkpoint runs/my_ppo/best.pt --viewer
        # 实时可视化窗口（需要显示器；鼠标拖动旋转视角、滚轮缩放，R 重置）
    python scripts/play_ppo.py --checkpoint runs/my_ppo/best.pt
        # 无头模式：离屏渲染录 mp4 视频

注意：这里直接取高斯策略的均值（确定性动作），不经过 take_action，
所以只要你的 PolicyNet.forward 输出动作均值即可。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load  # noqa: E402

PLAY_CFG = load("play")


def checkpoint_observation_dim(checkpoint_path):
    """从 Actor 第一层权重推断观测维度，兼容 45/48 维 checkpoint。"""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = checkpoint["actor"]
    first_weight = next(
        value for key, value in actor_state.items()
        if key.endswith("network.0.weight") or key.endswith("fc1.weight")
    )
    return int(first_weight.shape[1])


def parse_args():
    p = argparse.ArgumentParser(description="回放你自己的 PPO 四足狗策略")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--viewer", action="store_true", help="打开实时可视化窗口（默认录视频）")
    p.add_argument("--keyboard", action="store_true",
                   help="viewer 中用键盘控制速度指令（自动启用 viewer）")
    p.add_argument("--output", type=str, default=None, help="视频输出路径（viewer 模式忽略）")
    p.add_argument("--duration", type=float, default=None, help="默认 config/play.yaml")
    p.add_argument("--fps", type=int, default=None, help="默认 config/play.yaml")
    p.add_argument("--hidden-dim", type=int, default=None, help="默认 config/train.yaml（须与训练时一致）")
    return p.parse_args()


def command_at(t: float):
    command_script = PLAY_CFG["command_script"]
    total = sum(d for d, _ in command_script)
    t = t % total
    acc = 0.0
    for dur, cmd in command_script:
        if t < acc + dur:
            return cmd
        acc += dur
    return command_script[-1][1]


def load_agent(env, args):
    import torch
    from PPO import PPO

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = ckpt.get("config", {})
    saved_train = saved_config.get("train", saved_config)
    ppo_cfg = saved_train.get("ppo", load("train")["ppo"])
    hidden_dims = ([args.hidden_dim, args.hidden_dim] if args.hidden_dim is not None
                   else ppo_cfg["hidden_dims"])
    agent = PPO(
        state_dim=env.obs_dim, hidden_dim=hidden_dims[0], hidden_dims=hidden_dims,
        action_dim=env.act_dim, actor_lr=ppo_cfg["actor_lr"], critic_lr=ppo_cfg["critic_lr"],
        lmbda=ppo_cfg["lmbda"], epoch=ppo_cfg["epochs"], eps=ppo_cfg["eps"],
        batch_size=ppo_cfg["batch_size"], actor_gamma=ppo_cfg["gamma"],
        critic_gamma=ppo_cfg["gamma"], entropy_coef=ppo_cfg["entropy_coef"],
        value_coef=ppo_cfg["value_coef"], value_clip=ppo_cfg["value_clip"],
        max_grad_norm=ppo_cfg["max_grad_norm"], target_kl=ppo_cfg["target_kl"],
        activation=ppo_cfg["activation"], hidden_init_gain=ppo_cfg["hidden_init_gain"],
        actor_output_gain=ppo_cfg["actor_output_gain"],
        critic_output_gain=ppo_cfg["critic_output_gain"],
        initial_log_std=ppo_cfg["initial_log_std"], log_std_bounds=ppo_cfg["log_std_bounds"],
        numerical_epsilon=ppo_cfg["numerical_epsilon"], device="cpu",
    )
    agent.actor_net.load_state_dict(ckpt["actor"])
    agent.actor_net.eval()
    return agent


def get_action(agent, obs):
    """确定性动作：直接取高斯均值。"""
    return agent.take_action(obs, deterministic=PLAY_CFG["deterministic"])


def run_viewer(args, env, agent):
    """实时可视化窗口：真实时间回放，R 键重置，鼠标旋转视角。"""
    import mujoco.viewer

    keyboard_command = np.zeros(3)

    def on_key(key):
        if args.keyboard:
            _update_keyboard_command(key, keyboard_command, env)
        elif key in (ord("R"), ord("r")):
            env.reset()

    viewer = mujoco.viewer.launch_passive(
        env.model, env.data,
        key_callback=on_key,
    )
    if args.keyboard:
        print("键盘控制：↑/↓ 前后，←/→ 转向，Q/E 横移，空格停止，R 重置")
        print("请先点击 MuJoCo viewer 窗口使其获得键盘焦点。stage2 仅支持前进/停止；"
              "后退和转向需 stage3，横移需 stage4。")
    else:
        print("可视化窗口已打开：鼠标拖动=旋转视角，滚轮=缩放，R=重置")

    obs, _ = env.reset()
    step = 0
    while viewer.is_running():
        t = step * env.dt
        t0 = time.perf_counter()

        env.set_commands(*(keyboard_command if args.keyboard else command_at(t)))
        action = get_action(agent, obs)
        obs, _, terminated, truncated, info = env.step(action)
        viewer.sync()

        if terminated or truncated:
            print(f"t={t:.1f}s 回合结束，自动重置")
            obs, _ = env.reset()
        if step % PLAY_CFG["viewer_log_every_steps"] == 0:
            print(f"t={t:5.1f}s  指令=({info['commands'][0]:+.2f}, {info['commands'][1]:+.2f}, "
                  f"{info['commands'][2]:+.2f})")

        # 真实时间节奏（50 Hz 控制）
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, env.dt - elapsed))
        step += 1

    # 关闭窗口后等 viewer 线程完全退出，避免退出时 glfw 销毁竞争导致段错误
    viewer.close()
    while viewer.is_running():
        time.sleep(PLAY_CFG["viewer_close_poll_seconds"])
    time.sleep(PLAY_CFG["viewer_close_grace_seconds"])


def _update_keyboard_command(key, command, env):
    """处理一次键盘事件；独立函数便于无 GUI 单元测试。"""
    cfg = PLAY_CFG["keyboard"]
    keys = cfg["keys"]
    # mujoco.viewer 传入 GLFW 键码；方向键等特殊键用名称映射
    SPECIAL_KEYS = {"space": 32, "up": 265, "down": 264, "left": 263, "right": 262}

    def code(name):
        value = keys[name].lower()
        if value in SPECIAL_KEYS:
            return SPECIAL_KEYS[value]
        return ord(value)

    key = ord(chr(key).lower()) if 0 <= key < 128 and chr(key).isalpha() else key
    handled = True
    if key == code("forward"):
        command[0] += cfg["linear_step"]
    elif key == code("backward"):
        command[0] -= cfg["linear_step"]
    elif key == code("left"):
        command[1] += cfg["lateral_step"]
    elif key == code("right"):
        command[1] -= cfg["lateral_step"]
    elif key == code("yaw_left"):
        command[2] += cfg["yaw_step"]
    elif key == code("yaw_right"):
        command[2] -= cfg["yaw_step"]
    elif key == code("stop"):
        command[:] = 0.0
    elif key == code("reset"):
        env.reset()
    else:
        handled = False
    command[:] = np.clip(command,
                         [-cfg["max_vx"], -cfg["max_vy"], -cfg["max_yaw_rate"]],
                         [cfg["max_vx"], cfg["max_vy"], cfg["max_yaw_rate"]])
    env.set_commands(*command)
    if handled and cfg.get("print_commands", True):
        key_name = {32: "SPACE", 265: "↑", 264: "↓", 263: "←", 262: "→"}.get(key)
        if key_name is None:
            try:
                key_name = chr(key).upper()
            except (TypeError, ValueError):
                key_name = str(key)
        print(f"按键={key_name} -> 指令 vx={command[0]:+.2f}, "
              f"vy={command[1]:+.2f}, yaw={command[2]:+.2f}")


def record_video(args, env, agent):
    import os

    os.environ.setdefault("MUJOCO_GL", PLAY_CFG["video"]["mujoco_gl"])

    import imageio
    import mujoco

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else \
        ROOT / "videos" / f"{ckpt_path.parent.name}_{ckpt_path.stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_cfg = PLAY_CFG["video"]
    renderer = mujoco.Renderer(env.model, height=video_cfg["height"], width=video_cfg["width"])
    writer = imageio.get_writer(out_path, fps=args.fps, codec=video_cfg["codec"])

    obs, _ = env.reset()
    total_steps = int(args.duration * args.fps)
    t_start = time.time()

    for step in range(total_steps):
        t = step / args.fps
        env.set_commands(*command_at(t))
        action = get_action(agent, obs)
        obs, _, terminated, truncated, info = env.step(action)

        renderer.update_scene(env.data, camera=video_cfg["camera"])
        writer.append_data(renderer.render())

        if terminated or truncated:
            print(f"t={t:.1f}s 回合结束，重置")
            obs, _ = env.reset()

        if step % args.fps == 0:
            from env.quadruped_env import _quat_rotate_inverse
            q = env.data.qpos[3:7]
            vel_base = _quat_rotate_inverse(q, env.data.qvel[0:3])
            print(f"t={t:5.1f}s  指令=({info['commands'][0]:+.2f}, {info['commands'][1]:+.2f}, "
                  f"{info['commands'][2]:+.2f})  实际=({vel_base[0]:+.2f}, {vel_base[1]:+.2f}, "
                  f"{env.data.qvel[5]:+.2f})  位置=({env.data.qpos[0]:+.2f}, {env.data.qpos[1]:+.2f})")

    writer.close()
    print(f"视频已保存: {out_path}  （{total_steps} 帧，用时 {time.time() - t_start:.0f}s）")


def main():
    args = parse_args()
    if args.keyboard:
        args.viewer = True
    args.duration = PLAY_CFG["duration"] if args.duration is None else args.duration
    args.fps = PLAY_CFG["fps"] if args.fps is None else args.fps

    from env.quadruped_env import QuadrupedEnv

    checkpoint_obs_dim = checkpoint_observation_dim(args.checkpoint)
    if checkpoint_obs_dim not in (45, 48, 54, 57):
        raise ValueError(f"不支持的 checkpoint 观测维度: {checkpoint_obs_dim}")
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_pd = checkpoint.get("config", {}).get("pd")
    saved_env = checkpoint.get("config", {}).get("env")
    env = QuadrupedEnv(seed=PLAY_CFG["seed"], add_noise=PLAY_CFG["add_noise"],
                       randomize=PLAY_CFG["randomize"],
                       command_override=PLAY_CFG["command_override"],
                       include_base_lin_vel=checkpoint_obs_dim in (48, 54, 57),
                       include_gait_obs=checkpoint_obs_dim in (54, 57),
                       include_heading_obs=checkpoint_obs_dim == 57,
                       config_override=saved_env,
                       pd_config_override=saved_pd)
    # 回放必须使用训练时的残差尺度；旧 checkpoint 从课程等级推导
    saved_residual = checkpoint.get("curriculum_residual_scale")
    if saved_residual is None:
        levels = checkpoint.get("config", {}).get("train", {}).get("curriculum", {}).get("levels")
        level = (checkpoint.get("curriculum_state") or {}).get("level")
        if levels and level is not None:
            saved_residual = levels[int(level)].get("residual_scale")
    if saved_residual is not None:
        env.set_curriculum(0.0, 0.0, residual_scale=float(saved_residual))
    agent = load_agent(env, args)
    print(f"已加载 {args.checkpoint}（{checkpoint_obs_dim} 维观测）")

    if args.viewer:
        run_viewer(args, env, agent)
    else:
        record_video(args, env, agent)


if __name__ == "__main__":
    main()
