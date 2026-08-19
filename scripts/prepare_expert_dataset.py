"""Retarget DogML walking clips and replay them in MuJoCo to build state-action data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load  # noqa: E402
from env.quadruped_env import DEFAULT_DOF_POS, DOF_LOWER, DOF_UPPER, QuadrupedEnv  # noqa: E402


def recover_keypoints(data, joints=16):
    """Recover DogML root-invariant coordinates without importing its GUI visualizer."""
    rot_vel = data[:, 0]
    angle = np.zeros_like(rot_vel)
    angle[1:] = np.cumsum(rot_vel[:-1])
    quat = np.zeros((len(data), 4))
    quat[:, 0], quat[:, 2] = np.cos(angle), np.sin(angle)
    position = np.zeros((len(data), 3))
    position[1:, [0, 2]] = data[:-1, 1:3]
    qvec = -quat[:, 1:]
    uv = np.cross(qvec, position)
    uuv = np.cross(qvec, uv)
    position = position + 2 * (quat[:, :1] * uv + uuv)
    position = np.cumsum(position, axis=0)
    position[:, 1] = data[:, 3]
    local = data[:, 4:(joints - 1) * 3 + 4].reshape(len(data), joints - 1, 3)
    q = np.repeat(np.c_[quat[:, :1], -quat[:, 1:]][:, None, :], joints - 1, axis=1)
    qv = q[..., 1:]
    uv = np.cross(qv, local)
    local = local + 2 * (q[..., :1] * uv + np.cross(qv, uv))
    local[..., 0] += position[:, None, 0]
    local[..., 2] += position[:, None, 2]
    return np.concatenate([position[:, None], local], axis=1)


def retarget_ik(points, cfg):
    chains = cfg["chains"]
    order = cfg["leg_chain_order"]
    hip_i, foot_i = cfg["hip_point_in_chain"], cfg["foot_point_in_chain"]
    l1, l2 = cfg["link_lengths"]
    targets = np.zeros((len(points), 12))
    for target_leg, source_chain in enumerate(order):
        chain = chains[source_chain]
        vector = points[:, chain[foot_i]] - points[:, chain[hip_i]]
        x = cfg["forward_sign"] * vector[:, cfg["source_forward_axis"]]
        y = cfg["lateral_sign"] * vector[:, cfg["source_lateral_axis"]]
        z = vector[:, cfg["source_up_axis"]]
        radius = np.sqrt(x * x + y * y + z * z)
        scale = np.minimum((l1 + l2 - 1e-3) / np.maximum(radius, 1e-6), 1.0)
        x, y, z = x * scale, y * scale, z * scale
        ab = np.arctan2(y, -z)
        sagittal_z = -np.sqrt(np.maximum(y * y + z * z, 1e-8))
        r2 = x * x + sagittal_z * sagittal_z
        cos_knee = np.clip((r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
        knee = -np.arccos(cos_knee)
        hip = np.arctan2(-x, -sagittal_z) - np.arctan2(
            l2 * np.sin(knee), l1 + l2 * np.cos(knee))
        targets[:, target_leg * 3:(target_leg + 1) * 3] = np.c_[ab, hip, knee]
    margin = float(cfg["joint_margin"])
    targets = np.clip(targets, DOF_LOWER + margin, DOF_UPPER - margin)
    window = int(cfg["smoothing_window"])
    if window > 1:
        kernel = np.ones(window) / window
        targets = np.stack([np.convolve(targets[:, i], kernel, mode="same")
                            for i in range(12)], axis=1)
        targets[:window] = targets[window]
        targets[-window:] = targets[-window - 1]
    return targets


def labels_for(path, text_dir):
    text_path = Path(text_dir) / f"{path.stem}.txt"
    if not text_path.exists():
        return set()
    return set(text_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
               .lstrip("#").replace("#", ",").split(","))


def main():
    cfg = load("expert")
    rng = np.random.default_rng(cfg["seed"])
    files = [p for p in Path(cfg["motion_dir"]).glob("*.npy")
             if labels_for(p, cfg["text_dir"]) & set(cfg["labels"])]
    rng.shuffle(files)
    files = files[:int(cfg["max_clips"])]
    states, actions, accepted, rejected = [], [], 0, []
    replay = cfg["replay"]
    env_override = {"gait": {"reference": {"enabled": False},
                             "residual_control": {"enabled": False}}}
    pd_override = {"action_scale": replay["action_scale"]}
    stride = cfg["source_fps"] / cfg["target_fps"]
    for path in files:
        raw = np.load(path, allow_pickle=False)
        points = recover_keypoints(raw)
        indices = np.minimum((np.arange(int(len(raw) / stride)) * stride).astype(int), len(raw) - 1)
        targets = retarget_ik(points[indices], cfg["retarget"])
        root_forward = -np.diff(points[:, 0, cfg["retarget"]["source_forward_axis"]])
        command = float(np.clip(np.median(root_forward) * cfg["source_fps"],
                                replay["command_min"], replay["command_max"]))
        env = QuadrupedEnv(seed=cfg["seed"], add_noise=False, randomize=False,
                           command_override=True, config_override=env_override,
                           pd_config_override=pd_override)
        obs, _ = env.reset()
        env.set_commands(command, 0.0, 0.0)
        clip_states, clip_actions, body_contacts, yaw_rates, forward_velocities = [], [], 0, [], []
        terminated = False
        for frame, target in enumerate(targets):
            action = np.clip((target - DEFAULT_DOF_POS) / replay["action_scale"], -1.0, 1.0)
            next_obs, _, terminated, truncated, info = env.step(action)
            if frame >= replay["warmup_frames"]:
                clip_states.append(obs)
                clip_actions.append(action.astype(np.float32))
                body_contacts += info["body_contact_count"] + info["torso_contact_count"]
                yaw_rates.append(abs(info["base_ang_vel"][2]))
                forward_velocities.append(info["base_lin_vel"][0])
            obs = next_obs
            if terminated or truncated:
                break
        survival = (frame + 1) / len(targets)
        valid = (survival >= replay["minimum_survival_fraction"] and
                 len(clip_states) >= replay["minimum_clip_samples"] and
                 body_contacts <= replay["maximum_body_contacts"] and yaw_rates and
                 float(np.mean(yaw_rates)) <= replay["maximum_abs_yaw_rate"] and
                 abs(float(np.mean(forward_velocities)) - command) <=
                 replay["maximum_velocity_error"])
        if valid:
            states.extend(clip_states)
            actions.extend(clip_actions)
            accepted += 1
        else:
            rejected.append({"file": path.name, "survival": survival,
                             "body_contacts": body_contacts,
                             "samples": len(clip_states),
                             "velocity_error": (abs(float(np.mean(forward_velocities)) - command)
                                                if forward_velocities else None),
                             "mean_abs_yaw_rate": float(np.mean(yaw_rates)) if yaw_rates else None})
        if len(states) >= replay["maximum_samples"]:
            break
    report = {"considered": len(files), "accepted_clips": accepted,
              "samples": len(states), "rejected": rejected}
    output = ROOT / cfg["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["report_path"]).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    if not states:
        raise SystemExit("没有 DogML 轨迹通过严格 MuJoCo 复演门槛；未覆盖专家数据文件")
    np.savez_compressed(output, states=np.asarray(states, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32))
    print(f"专家数据: {len(states)} samples, {accepted}/{len(files)} clips -> {output}")


if __name__ == "__main__":
    main()
