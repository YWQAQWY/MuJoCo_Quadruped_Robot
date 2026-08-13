"""用你自己的 PPO.py 训练出的策略回放并录视频。

用法：
    .venv/bin/python scripts/play_ppo.py --checkpoint runs/my_ppo/best.pt

注意：这里直接取高斯策略的均值（确定性动作），不经过 take_action，
所以只要你的 PolicyNet.forward 输出动作均值即可。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # 无头渲染，必须在 import mujoco 前设置

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.quadruped_env import QuadrupedEnv, _quat_rotate_inverse  # noqa: E402
from PPO import PPO  # noqa: E402

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
    p = argparse.ArgumentParser(description="回放你自己的 PPO 四足狗策略并录视频")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--hidden-dim", type=int, default=256)
    return p.parse_args()


def command_at(t: float):
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
    out_path = Path(args.output) if args.output else \
        ROOT / "videos" / f"{ckpt_path.parent.name}_{ckpt_path.stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = QuadrupedEnv(seed=0, add_noise=False, randomize=False, command_override=True)

    agent = PPO(state_dim=env.obs_dim, hidden_dim=args.hidden_dim, action_dim=env.act_dim,
                actor_lr=1e-3, critic_lr=1e-3, lmbda=0.95, epoch=5, eps=0.2,
                actor_gamma=0.99, critic_gamma=0.99, device="cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent.actor_net.load_state_dict(ckpt["actor"])
    agent.actor_net.eval()
    print(f"已加载 {ckpt_path}")

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264")

    obs, _ = env.reset()
    total_steps = int(args.duration * args.fps)
    t_start = time.time()

    for step in range(total_steps):
        t = step / args.fps
        env.set_commands(*command_at(t))

        # 确定性动作：直接取高斯均值
        with torch.no_grad():
            obs_t = torch.tensor([obs], dtype=torch.float)
            action = agent.actor_net(obs_t).squeeze(0).numpy()

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
