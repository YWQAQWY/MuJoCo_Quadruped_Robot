"""Scan the configured open-loop trot and reject PPO training if physics cannot walk."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load  # noqa: E402
from env.quadruped_env import QuadrupedEnv  # noqa: E402


def run_candidate(scan_cfg, parameters):
    env_override = {
        "gait": {
            "phase_frequency": parameters["phase_frequency"],
            "duty_factor": parameters["duty_factor"],
            "randomize_initial_phase": False,
            "residual_control": {"enabled": True, "scale": 0.0},
            "reference": {
                "enabled": True,
                **scan_cfg["reference_scaling"],
                "joint_amplitudes": [0.0, parameters["hip_pitch_amplitude"],
                                     parameters["knee_amplitude"]],
                "swing_knee_lift": parameters["swing_knee_lift"],
                "phase_bias": parameters["phase_bias"],
            },
        },
        "domain_randomization": {"push": {"probability": 0.0}},
    }
    env = QuadrupedEnv(seed=scan_cfg["seed"], add_noise=False, randomize=False,
                       command_override=True, config_override=env_override)
    obs, _ = env.reset()
    del obs
    env.set_commands(*scan_cfg["command"])
    steps = int(round(scan_cfg["duration_seconds"] / env.dt))
    velocities, contacts, torso_contacts = [], [], 0
    terminated = False
    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(np.zeros(env.act_dim))
        velocities.append(info["base_lin_vel"])
        contacts.append(info["foot_contacts"])
        torso_contacts += info["torso_contact_count"]
        if terminated or truncated:
            break
    contacts = np.asarray(contacts, dtype=bool)
    mean_velocity = np.mean(velocities, axis=0)
    result = {
        "parameters": parameters,
        "reference_scaling": {
            key: env.cfg["gait"]["reference"][key] for key in (
                "nominal_speed", "minimum_speed_scale", "maximum_speed_scale",
                "minimum_lift_scale", "maximum_lift_scale")
        },
        "forward_velocity": float(mean_velocity[0]),
        "survival_fraction": len(velocities) / steps,
        "foot_air_rates": np.mean(~contacts, axis=0).tolist(),
        "all_feet_contact_rate": float(np.mean(np.all(contacts, axis=1))),
        "torso_contacts": int(torso_contacts),
        "terminated": bool(terminated),
    }
    result["qualified"] = bool(
        result["forward_velocity"] >= scan_cfg["minimum_forward_velocity"] and
        result["survival_fraction"] >= scan_cfg["minimum_survival_fraction"] and
        result["torso_contacts"] <= scan_cfg["maximum_torso_contacts"] and
        result["all_feet_contact_rate"] <= scan_cfg["maximum_all_feet_contact_rate"] and
        min(result["foot_air_rates"]) >= scan_cfg["minimum_air_rate_per_foot"]
    )
    return result


def main():
    cfg = load("gait_scan")
    grid = cfg["grid"]
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[key] for key in keys)):
        parameters = dict(zip(keys, values))
        result = run_candidate(cfg, parameters)
        results.append(result)
        print(f"vx={result['forward_velocity']:+.3f} survival={result['survival_fraction']:.2f} "
              f"air_min={min(result['foot_air_rates']):.2f} params={parameters}")
    results.sort(key=lambda item: (item["qualified"], item["forward_velocity"]), reverse=True)
    report_path = ROOT / cfg["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    qualified = [item for item in results if item["qualified"]]
    if not qualified:
        raise SystemExit(f"没有参考步态通过硬门槛；禁止开始 PPO。报告: {report_path}")
    print(f"通过 {len(qualified)}/{len(results)} 组；最佳参数: {qualified[0]['parameters']}")
    print(f"报告: {report_path}")

    profile_cfg = cfg["profile_validation"]
    stage_env = load("stages")["stage2_forward"]["env"]
    profile_results = []
    for command in profile_cfg["commands"]:
        env = QuadrupedEnv(seed=cfg["seed"], add_noise=False, randomize=False,
                           command_override=True, config_override=stage_env)
        env.reset()
        env.set_commands(*command)
        velocities, terminated = [], False
        steps = int(round(cfg["duration_seconds"] / env.dt))
        for _ in range(steps):
            _, _, terminated, truncated, info = env.step(np.zeros(env.act_dim))
            velocities.append(info["base_lin_vel"])
            if terminated or truncated:
                break
        velocity = np.mean(velocities, axis=0)
        error = abs(float(command[0]) - float(velocity[0]))
        passed = bool(error <= profile_cfg["maximum_linear_error"] and
                      abs(velocity[1]) <= profile_cfg["maximum_lateral_speed"] and
                      abs(velocity[2]) <= profile_cfg["maximum_yaw_rate"] and
                      len(velocities) / steps >= profile_cfg["minimum_survival_fraction"])
        profile_results.append({"command": command, "actual_velocity": velocity.tolist(),
                                "linear_error": error,
                                "survival_fraction": len(velocities) / steps,
                                "qualified": passed})
    profile_path = ROOT / cfg["profile_report_path"]
    profile_path.write_text(json.dumps(profile_results, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    if not all(item["qualified"] for item in profile_results):
        raise SystemExit(f"参考步态速度映射未通过全指令验证: {profile_path}")
    print(f"全指令速度映射验证通过: {profile_path}")


if __name__ == "__main__":
    main()
