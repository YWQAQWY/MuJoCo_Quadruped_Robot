"""PD 关节控制器：把策略动作转换成电机力矩。

链路（见 README「PPO → PD Control Chain」）：
    action (12-d, 已裁剪到 [-1,1])
      → q_target = q_default + action_scale · action     (set_action)
      → τ = kp · (q_target − q) − kd · q̇                 (compute_torques)
      → clip(τ, ±torque_limit) → 写入 data.ctrl（XML 中 12 个 <motor> 的力矩）

本模块只负责「角度 → 力矩」，推进物理（mj_step）由环境完成。

超参数（kp/kd/力矩限幅/动作缩放）在 config/pd.yaml。
"""
from __future__ import annotations

import numpy as np

from config import deep_merge, load

PD_CFG = load("pd")  # config/pd.yaml


class PDController:
    def __init__(
        self,
        data,
        default_dof_pos: np.ndarray,
        joint_qpos_adr: int,
        joint_qvel_adr: int,
        kp: float | None = None,
        kd: float | None = None,
        torque_limit: float | None = None,
        action_scale: float | None = None,
        config_override: dict | None = None,
    ):
        self.data = data
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=float)
        self.num_joints = len(self.default_dof_pos)
        self.joint_qpos_adr = joint_qpos_adr
        self.joint_qvel_adr = joint_qvel_adr
        cfg = deep_merge(PD_CFG, config_override)
        # 显式传参优先，否则用合并后的 PD 配置
        self.kp = float(cfg["kp"] if kp is None else kp)
        self.kd = float(cfg["kd"] if kd is None else kd)
        self.torque_limit = float(cfg["torque_limit"] if torque_limit is None else torque_limit)
        self.action_scale = float(cfg["action_scale"] if action_scale is None else action_scale)
        self.target = self.default_dof_pos.copy()

    def set_action(self, action: np.ndarray) -> np.ndarray:
        """策略动作 → 目标关节角。action: [num_joints]，会被裁剪到 [-1, 1]。"""
        action = np.clip(action, -1.0, 1.0)
        self.target = self.default_dof_pos + self.action_scale * action
        return self.target

    def set_target_offsets(self, offsets: np.ndarray) -> np.ndarray:
        """Set joint targets from radian offsets relative to the default pose."""
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (self.num_joints,):
            raise ValueError(f"offsets shape must be {(self.num_joints,)}, got {offsets.shape}")
        self.target = self.default_dof_pos + offsets
        return self.target

    def compute_torques(self) -> np.ndarray:
        """根据当前关节状态计算 PD 力矩（已限幅），不推进物理。"""
        q = self.data.qpos[self.joint_qpos_adr : self.joint_qpos_adr + self.num_joints]
        dq = self.data.qvel[self.joint_qvel_adr : self.joint_qvel_adr + self.num_joints]
        tau = self.kp * (self.target - q) - self.kd * dq
        return np.clip(tau, -self.torque_limit, self.torque_limit)
