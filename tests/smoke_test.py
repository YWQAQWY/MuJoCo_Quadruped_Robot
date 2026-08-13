"""冒烟测试：环境能跑、形状正确、PD 响应方向正确。

用法：.venv/bin/python tests/smoke_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.quadruped_env import DEFAULT_DOF_POS, QuadrupedEnv  # noqa: E402


def test_shapes_and_random_steps():
    env = QuadrupedEnv(seed=0)
    obs, _ = env.reset()
    assert obs.shape == (env.obs_dim,), f"观测形状错误: {obs.shape}"
    assert np.isfinite(obs).all(), "观测出现 NaN/Inf"

    t0 = time.time()
    for i in range(500):
        action = np.random.uniform(-1, 1, env.act_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (env.obs_dim,)
        assert np.isfinite(reward) and np.isfinite(obs).all(), f"第 {i} 步出现 NaN"
        if terminated or truncated:
            obs, _ = env.reset()
    fps = 500 / (time.time() - t0)
    print(f"[OK] 随机动作 500 步无异常，仿真速度 {fps:.0f} 步/s（控制频率）")
    print(f"     奖励分量: " + ", ".join(f"{k}={v:+.3f}" for k, v in info["reward_components"].items()))


def test_pd_response():
    """单个膝关节偏离默认值 0.1 rad 时，PD 应把它往回拉。"""
    env = QuadrupedEnv(seed=0, add_noise=False, randomize=False)
    env.reset()
    knee_idx = 2  # FL_knee
    env.data.qpos[env.joint_qpos_adr + knee_idx] = DEFAULT_DOF_POS[knee_idx] + 0.1
    env.data.qvel[:] = 0.0
    env.step(np.zeros(env.act_dim))
    q_after = env.data.qpos[env.joint_qpos_adr + knee_idx]
    error_after = abs(q_after - DEFAULT_DOF_POS[knee_idx])
    assert error_after < 0.1, f"PD 没有把膝关节拉回默认值: 误差 {error_after:.4f} rad"
    # 力矩符号：tau = kp*(target - q) < 0（往回拉的负力矩）
    assert env.data.ctrl[knee_idx] < 0, f"力矩方向错误: {env.data.ctrl[knee_idx]}"
    print(f"[OK] PD 响应正确：膝误差 0.1 rad → 一步后 {error_after:.4f} rad，力矩为负（回拉）")


def test_standing_contacts_are_not_penalized():
    env = QuadrupedEnv(seed=0, add_noise=False, randomize=False)
    env.reset()
    for _ in range(100):
        _, _, terminated, _, info = env.step(np.zeros(env.act_dim))
        assert not terminated
    assert info["reward_components"]["undesired_contact"] == 0.0


if __name__ == "__main__":
    test_shapes_and_random_steps()
    test_pd_response()
    test_standing_contacts_are_not_penalized()
    print("全部通过")
