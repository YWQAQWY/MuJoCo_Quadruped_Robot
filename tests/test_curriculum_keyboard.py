"""课程学习调度与键盘命令的无 GUI 回归测试。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import deep_merge, load  # noqa: E402
from env.quadruped_env import QuadrupedEnv  # noqa: E402
from scripts.play_ppo import _update_keyboard_command  # noqa: E402
from PPO import PPO  # noqa: E402
from scripts.train_ppo import (curriculum_scales, curriculum_values, initial_curriculum_state,
                               evaluate, load_state_with_input_expansion,
                               update_competence_curriculum)  # noqa: E402


def test_curriculum_schedule():
    cfg = load("train")["curriculum"]
    start = curriculum_scales(0, 100, cfg)
    end = curriculum_scales(99, 100, cfg)
    assert np.allclose(start, [cfg["command_scale_start"], cfg["randomization_scale_start"]])
    assert np.allclose(end, [1.0, 1.0])

    stage = load("stages")["stage2_forward"]
    stage_cfg = deep_merge(load("train"), stage["train"])["curriculum"]
    stage_state = initial_curriculum_state(stage_cfg)
    stage_start = curriculum_values(0, 500, stage_cfg, stage_state)
    assert np.allclose(stage_start, [0.0, 0.0, 0.15, 1.0])

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
    assert obs.shape == (54,)
    # 末尾为 phase sin/cos + FL/FR/RL/RR 接触状态。
    assert obs[-6:].shape == (6,)
    env.set_commands(0.5, 0.0, 0.0)
    _, _, _, _, info = env.step(np.zeros(env.act_dim))
    assert info["reward_components"]["no_motion"] < 0.0


def test_checkpoint_input_expansion():
    old = PPO(48, 16, 12, 3e-4, 1e-3, 0.95, 1, 0.2, 8, 0.99, 0.99, "cpu")
    new = PPO(54, 16, 12, 3e-4, 1e-3, 0.95, 1, 0.2, 8, 0.99, 0.99, "cpu")
    old_weight = old.actor_net.state_dict()["network.0.weight"].clone()
    assert load_state_with_input_expansion(new.actor_net, old.actor_net.state_dict())
    new_weight = new.actor_net.state_dict()["network.0.weight"]
    assert np.allclose(new_weight[:, :48].numpy(), old_weight.numpy())
    assert np.allclose(new_weight[:, 48:].numpy(), 0.0)


def test_reference_residual_and_gait_reward():
    stage_env = load("stages")["stage2_forward"]["env"]
    env = QuadrupedEnv(seed=3, add_noise=False, randomize=False,
                       command_override=True, config_override=stage_env)
    env.reset()
    env.set_commands(0.0, 0.0, 0.0)
    assert np.allclose(env._get_reference_joint_offsets(), 0.0)
    env.set_commands(0.3, 0.0, 0.0)
    assert not np.allclose(env._get_reference_joint_offsets(), 0.0)

    env.gait_phase = 0.25
    forces = np.full(4, env.cfg["gait"]["contact_force_normalization"])
    all_down = env._gait_style_terms(np.ones(4, dtype=bool), forces)["contact"]
    correct = np.array([True, False, False, True])
    correct_score = env._gait_style_terms(correct, forces * correct)["contact"]
    wrong = ~correct
    wrong_score = env._gait_style_terms(wrong, forces * wrong)["contact"]
    assert all_down <= 0.0
    assert correct_score > wrong_score


def test_competence_curriculum_requires_consecutive_passes():
    cfg = load("stages")["stage2_forward"]["train"]["curriculum"]
    state = initial_curriculum_state(cfg)
    passed = {"survival_rate": 1.0, "success_rate": 1.0}
    assert not update_competence_curriculum(state, cfg, passed)
    assert update_competence_curriculum(state, cfg, passed)
    assert state["level"] == 1


def test_static_policy_cannot_pass_forward_evaluation():
    class ZeroAgent:
        @staticmethod
        def take_action(obs, deterministic=True):
            del obs, deterministic
            return np.zeros(12)

    cfg = load("train")["evaluation"]
    cfg.update({"commands": [[0.2, 0.0, 0.0]], "episodes": 1, "episode_length": 25})
    metrics = evaluate(ZeroAgent(), 9, cfg)
    assert metrics["success_rate"] == 0.0


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
    test_checkpoint_input_expansion()
    test_reference_residual_and_gait_reward()
    test_competence_curriculum_requires_consecutive_passes()
    test_static_policy_cannot_pass_forward_evaluation()
    test_keyboard_commands()
    print("[OK] 课程学习与键盘控制测试通过")
