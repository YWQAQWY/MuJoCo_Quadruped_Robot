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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="回放你自己的 PPO 四足狗策略")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--viewer", action="store_true", help="打开实时可视化窗口（默认录视频）")
    p.add_argument("--output", type=str, default=None, help="视频输出路径（viewer 模式忽略）")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=None, help="默认 config/train.yaml（须与训练时一致）")
    return p.parse_args()


# (时长 s, (vx, vy, yaw_rate))
COMMAND_SCRIPT = [
    (2.0, (0.0, 0.0, 0.0)),
    (4.0, (0.8, 0.0, 0.0)),
    (4.0, (0.5, 0.4, 0.0)),
    (4.0, (0.0, 0.0, 0.8)),
    (4.0, (0.5, 0.0, 0.0)),
    (2.0, (0.0, 0.0, 0.0)),
]


def command_at(t: float):
    total = sum(d for d, _ in COMMAND_SCRIPT)
    t = t % total
    acc = 0.0
    for dur, cmd in COMMAND_SCRIPT:
        if t < acc + dur:
            return cmd
        acc += dur
    return COMMAND_SCRIPT[-1][1]


def load_agent(env, args):
    import torch
    from config import load
    from PPO import PPO

    if args.hidden_dim is None:
        args.hidden_dim = load("train")["ppo"]["hidden_dim"]  # 须与训练时一致

    agent = PPO(state_dim=env.obs_dim, hidden_dim=args.hidden_dim, action_dim=env.act_dim,
                actor_lr=1e-3, critic_lr=1e-3, lmbda=0.95, epoch=5, eps=0.2, batch_size=256,
                actor_gamma=0.99, critic_gamma=0.99, device="cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent.actor_net.load_state_dict(ckpt["actor"])
    agent.actor_net.eval()
    return agent


def get_action(agent, obs):
    """确定性动作：直接取高斯均值。"""
    return agent.take_action(obs, deterministic=True)


def run_viewer(args, env, agent):
    """实时可视化窗口：真实时间回放，R 键重置，鼠标旋转视角。"""
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(
        env.model, env.data,
        key_callback=lambda key: _on_key(key, env),
    )
    print("可视化窗口已打开：鼠标拖动=旋转视角，滚轮=缩放，R=重置，ESC 或关闭窗口退出")

    obs, _ = env.reset()
    step = 0
    while viewer.is_running():
        t = step * env.dt
        t0 = time.perf_counter()

        env.set_commands(*command_at(t))
        action = get_action(agent, obs)
        obs, _, terminated, truncated, info = env.step(action)
        viewer.sync()

        if terminated:
            print(f"t={t:.1f}s 机器人摔倒，自动重置")
            obs, _ = env.reset()
        if step % 50 == 0:
            print(f"t={t:5.1f}s  指令=({info['commands'][0]:+.2f}, {info['commands'][1]:+.2f}, "
                  f"{info['commands'][2]:+.2f})")

        # 真实时间节奏（50 Hz 控制）
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, env.dt - elapsed))
        step += 1

    # 关闭窗口后等 viewer 线程完全退出，避免退出时 glfw 销毁竞争导致段错误
    viewer.close()
    while viewer.is_running():
        time.sleep(0.01)
    time.sleep(0.05)


def _on_key(key, env):
    if key in (82, 114):  # R / r
        env.reset()


def record_video(args, env, agent):
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")  # 无头渲染后端

    import imageio
    import mujoco

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output) if args.output else \
        ROOT / "videos" / f"{ckpt_path.parent.name}_{ckpt_path.stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264")

    obs, _ = env.reset()
    total_steps = int(args.duration * args.fps)
    t_start = time.time()

    for step in range(total_steps):
        t = step / args.fps
        env.set_commands(*command_at(t))
        action = get_action(agent, obs)
        obs, _, terminated, truncated, info = env.step(action)

        renderer.update_scene(env.data, camera="track")
        writer.append_data(renderer.render())

        if terminated:
            print(f"t={t:.1f}s 机器人摔倒，重置")
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

    from env.quadruped_env import QuadrupedEnv

    env = QuadrupedEnv(seed=0, add_noise=False, randomize=False, command_override=True)
    agent = load_agent(env, args)
    print(f"已加载 {args.checkpoint}")

    if args.viewer:
        run_viewer(args, env, agent)
    else:
        record_video(args, env, agent)


if __name__ == "__main__":
    main()
