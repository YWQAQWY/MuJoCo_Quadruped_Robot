"""课程学习调度与键盘命令的无 GUI 回归测试。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load  # noqa: E402
from env.quadruped_env import QuadrupedEnv  # noqa: E402
from scripts.play_ppo import _update_keyboard_command  # noqa: E402
from scripts.train_ppo import curriculum_scales  # noqa: E402


def test_curriculum_schedule():
    cfg = load("train")["curriculum"]
    start = curriculum_scales(0, 100, cfg)
    end = curriculum_scales(99, 100, cfg)
    assert np.allclose(start, [cfg["command_scale_start"], cfg["randomization_scale_start"]])
    assert np.allclose(end, [1.0, 1.0])

    env = QuadrupedEnv(seed=0, add_noise=False, randomize=False)
    env.set_curriculum(*start)
    start_ranges = env.cfg["commands"]["curriculum_start_ranges"]
    for _ in range(50):
        env._resample_commands(first=False)
        assert start_ranges["vx"][0] <= env.commands[0] <= start_ranges["vx"][1]
        assert env.commands[1] == 0.0 and env.commands[2] == 0.0


def test_velocity_observation_and_no_motion_penalty():
    env = QuadrupedEnv(seed=1, add_noise=False, randomize=False, command_override=True)
    obs, _ = env.reset()
    assert obs.shape == (48,)
    env.set_commands(0.5, 0.0, 0.0)
    _, _, _, _, info = env.step(np.zeros(env.act_dim))
    assert info["reward_components"]["no_motion"] < 0.0


def test_keyboard_commands():
    env = QuadrupedEnv(seed=0, add_noise=False, randomize=False, command_override=True)
    env.reset()
    command = np.zeros(3)
    _update_keyboard_command(ord("w"), command, env)
    assert command[0] > 0 and np.allclose(env.commands, command)
    _update_keyboard_command(ord("a"), command, env)
    _update_keyboard_command(ord("q"), command, env)
    assert command[1] > 0 and command[2] > 0
    _update_keyboard_command(32, command, env)
    assert np.allclose(command, 0.0)


if __name__ == "__main__":
    test_curriculum_schedule()
    test_velocity_observation_and_no_motion_penalty()
    test_keyboard_commands()
    print("[OK] 课程学习与键盘控制测试通过")
