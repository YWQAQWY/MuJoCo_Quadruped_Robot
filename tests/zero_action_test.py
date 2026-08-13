"""零动作站立测试：action=0（即 PD 目标=默认姿态）时机器人应稳定站立 5 秒。

用法：.venv/bin/python tests/zero_action_test.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.quadruped_env import QuadrupedEnv, _quat_to_rpy  # noqa: E402


def main():
    env = QuadrupedEnv(seed=1, add_noise=False, randomize=False)
    env.reset()

    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    min_base_z = 1e9
    terminated = False

    steps = 250  # 5 s
    for i in range(steps):
        _, _, terminated, truncated, info = env.step(np.zeros(env.act_dim))
        assert not terminated, f"第 {i} 步摔倒（零动作下应能站立）"
        rpy = _quat_to_rpy(env.data.qpos[3:7])
        max_abs_roll = max(max_abs_roll, abs(rpy[0]))
        max_abs_pitch = max(max_abs_pitch, abs(rpy[1]))
        min_base_z = min(min_base_z, env.data.qpos[2])

    print(f"[OK] 零动作站立 {steps * env.dt:.1f} s")
    print(f"     max|roll|={np.rad2deg(max_abs_roll):.2f}°  max|pitch|={np.rad2deg(max_abs_pitch):.2f}°  "
          f"最低基座高度={min_base_z:.3f} m")
    assert max_abs_roll < 0.1, f"roll 过大: {max_abs_roll}"
    assert max_abs_pitch < 0.1, f"pitch 过大: {max_abs_pitch}"
    assert min_base_z > 0.25, f"基座下沉过多: {min_base_z}"
    print("[OK] 通过：PD 增益与默认姿态可以稳定站立")


if __name__ == "__main__":
    main()
