"""PD 关节控制器：把策略动作转换成电机力矩。

链路（见 README「PPO → PD Control Chain」）：
    action (12-d, 已裁剪到 [-1,1])
      → q_target = q_default + action_scale · action     (set_action)
      → τ = kp · (q_target − q) − kd · q̇                 (compute_torques)
      → clip(τ, ±torque_limit) → 写入 data.ctrl（XML 中 12 个 <motor> 的力矩）

本模块只负责「角度 → 力矩」，推进物理（mj_step）由环境完成。
"""
from __future__ import annotations

import numpy as np

KP = 20.0            # 比例增益 (Nm/rad)
KD = 0.5             # 微分增益 (Nm·s/rad)
TORQUE_LIMIT = 33.5  # 力矩限幅 (Nm)
ACTION_SCALE = 0.25  # 动作缩放：动作 ±1 对应目标角偏移 ±0.25 rad


class PDController:
    def __init__(
        self,
        data,
        default_dof_pos: np.ndarray,
        joint_qpos_adr: int,
        joint_qvel_adr: int,
        kp: float = KP,
        kd: float = KD,
        torque_limit: float = TORQUE_LIMIT,
        action_scale: float = ACTION_SCALE,
    ):
        self.data = data
        self.default_dof_pos = np.asarray(default_dof_pos, dtype=float)
        self.num_joints = len(self.default_dof_pos)
        self.joint_qpos_adr = joint_qpos_adr
        self.joint_qvel_adr = joint_qvel_adr
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit
        self.action_scale = action_scale
        self.target = self.default_dof_pos.copy()

    def set_action(self, action: np.ndarray) -> np.ndarray:
        """策略动作 → 目标关节角。action: [num_joints]，会被裁剪到 [-1, 1]。"""
        action = np.clip(action, -1.0, 1.0)
        self.target = self.default_dof_pos + self.action_scale * action
        return self.target

    def compute_torques(self) -> np.ndarray:
        """根据当前关节状态计算 PD 力矩（已限幅），不推进物理。"""
        q = self.data.qpos[self.joint_qpos_adr : self.joint_qpos_adr + self.num_joints]
        dq = self.data.qvel[self.joint_qvel_adr : self.joint_qvel_adr + self.num_joints]
        tau = self.kp * (self.target - q) - self.kd * dq
        return np.clip(tau, -self.torque_limit, self.torque_limit)
