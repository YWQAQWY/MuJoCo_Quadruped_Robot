"""用训练好的策略回放并录制视频（离屏渲染，无需显示器）。

用法：
    python scripts/play.py --checkpoint runs/my_run/best.pt
    python scripts/play.py --checkpoint runs/my_run/best.pt --duration 20 --output videos/walk.mp4

指令序列（可自行修改 COMMAND_SCRIPT）：
    静止 2s → 直行 0.8 m/s → 斜行 → 原地转向 → 直行 → 静止
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # 无头渲染后端，必须在 import mujoco 前设置

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from env.quadruped_env import QuadrupedEnv, _quat_rotate_inverse  # noqa: E402
from ppo.agent import PPO  # noqa: E402

# (时长 s, (vx, vy, yaw_rate))
COMMAND_SCRIPT = [
    (2.0, (0.0, 0.0, 0.0)),
    (4.0, (0.8, 0.0, 0.0)),
    (4.0, (0.5, 0.4, 0.0)),
    (4.0, (0.0, 0.0, 0.8)),
    (4.0, (0.5, 0.0, 0.0)),
    (2.0, (0.0, 0.0, 0.0)),
]


def parse_args():
    p = argparse.ArgumentParser(description="回放训练好的四足狗策略并录视频")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default=None, help="默认 videos/<模型名>.mp4")
    p.add_argument("--duration", type=float, default=20.0, help="回放时长（秒），超过指令脚本则重复")
    p.add_argument("--fps", type=int, default=50, help="视频帧率（= 控制频率）")
    return p.parse_args()


def command_at(t: float) -> tuple[float, float, float]:
    total = sum(d for d, _ in COMMAND_SCRIPT)
    t = t % total
    acc = 0.0
    for dur, cmd in COMMAND_SCRIPT:
        if t < acc + dur:
            return cmd
        acc += dur
    return COMMAND_SCRIPT[-1][1]


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ROOT / "videos" / f"{ckpt_path.parent.name}_{ckpt_path.stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = QuadrupedEnv(seed=0, add_noise=False, randomize=False, command_override=True)
    agent = PPO(obs_dim=env.obs_dim, act_dim=env.act_dim, device="cpu")
    agent.load(ckpt_path, load_optimizer=False)
    agent.eval_mode()
    print(f"已加载 {ckpt_path}")

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264")

    obs, _ = env.reset()
    mujoco.mj_forward(env.model, env.data)
    total_steps = int(args.duration * args.fps)
    t_start = time.time()

    for step in range(total_steps):
        t = step / args.fps
        env.set_commands(*command_at(t))

        action, _, _ = agent.act(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        renderer.update_scene(env.data, camera="track")
        writer.append_data(renderer.render())

        if terminated:
            print(f"t={t:.1f}s 机器人摔倒，重置")
            obs, _ = env.reset()

        if step % args.fps == 0:
            q = env.data.qpos[3:7]
            vel_base = _quat_rotate_inverse(q, env.data.qvel[0:3])
            print(f"t={t:5.1f}s  指令=({info['commands'][0]:+.2f}, {info['commands'][1]:+.2f}, "
                  f"{info['commands'][2]:+.2f})  实际=({vel_base[0]:+.2f}, {vel_base[1]:+.2f}, "
                  f"{env.data.qvel[5]:+.2f})  位置=({env.data.qpos[0]:+.2f}, {env.data.qpos[1]:+.2f})")

    writer.close()
    print(f"视频已保存: {out_path}  （{total_steps} 帧，用时 {time.time() - t_start:.0f}s）")


if __name__ == "__main__":
    main()
