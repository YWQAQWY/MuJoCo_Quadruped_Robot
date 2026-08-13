"""四足机械狗 MuJoCo 环境（完整实现，无留空）。

接口风格对齐 gymnasium：
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

控制方式：策略输出 12 维动作（目标关节角偏移），环境内手写 PD 计算力矩：
    tau = kp * (target - q) - kd * dq
模拟频率 200 Hz（dt=0.005），控制频率 50 Hz（decimation=4），回合 1000 步 = 20 s。
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

XML_PATH = Path(__file__).resolve().parent.parent / "robot" / "quadruped.xml"

# 关节顺序 = XML 中声明顺序（FL FR RL RR，每条腿 ab / pitch / knee）
# 默认姿态参照 Unitree A1：前腿 hip_pitch 0.8、后腿 1.0、膝 -1.5，
# hip_ab ±0.1（足端外撇），支撑多边形前后 0.44 m、左右 0.29 m，站立更稳
DEFAULT_DOF_POS = np.array([
    0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5,
])
DOF_LOWER = np.array([-0.5, -1.0, -2.9] * 4)
DOF_UPPER = np.array([0.5, 1.4, 0.0] * 4)

KP = 20.0          # PD 比例增益 (Nm/rad)
KD = 0.5           # PD 微分增益 (Nm·s/rad)
TORQUE_LIMIT = 33.5
ACTION_SCALE = 0.25
BASE_HEIGHT_TARGET = 0.37   # 默认姿态下躯干中心的标称高度（由 XML 几何算出）
FALL_HEIGHT = 0.18          # 躯干低于此高度判定摔倒
COMMAND_RESAMPLE_STEPS = 500  # 每 10 s 重采样一次指令


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 q=[w,x,y,z] 旋转向量 v。"""
    w, x, y, z = q
    qv = q[1:]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q 的逆旋转：把世界系向量转到机体坐标系。"""
    return _quat_rotate(np.array([q[0], -q[1], -q[2], -q[3]]), v)


def _quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] -> 欧拉角 [roll, pitch, yaw]。"""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class QuadrupedEnv:
    def __init__(
        self,
        xml_path: Path = XML_PATH,
        seed: int | None = None,
        episode_length: int = 1000,
        decimation: int = 4,
        add_noise: bool = True,       # 观测噪声（训练开、评估关）
        randomize: bool = True,       # 域随机化（训练开、评估关）
        command_override: bool = False,  # 评估时由外部脚本设定指令
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.episode_length = episode_length
        self.decimation = decimation
        self.add_noise = add_noise
        self.randomize = randomize
        self.command_override = command_override
        self.dt = self.model.opt.timestep * decimation  # 控制周期 0.02 s

        self.obs_dim = 45
        self.act_dim = 12
        self.num_joints = 12
        self.step_count = 0

        # 运动学/动力学索引
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")
            for leg in ("FL", "FR", "RL", "RR")
        ]
        self.joint_qpos_adr = 7   # 自由关节占 7 个 qpos
        self.joint_qvel_adr = 6   # 自由关节占 6 个 qvel

        # 观测缩放（把不同量纲的量压到相近尺度，legged_gym 惯例）
        self.obs_scales = np.concatenate([
            np.array([0.25] * 3),           # base 角速度 (rad/s)
            np.array([1.0] * 3),            # 投影重力
            np.array([2.0, 2.0, 0.25]),     # 指令 vx, vy, yaw
            np.array([1.0] * 12),           # 关节角 - 默认值
            np.array([0.05] * 12),          # 关节角速度
            np.array([1.0] * 12),           # 上一步动作
        ])

        self.commands = np.zeros(3)         # [vx, vy, yaw_rate]，机体坐标系
        self.actions = np.zeros(self.act_dim)
        self.last_action = np.zeros(self.act_dim)
        self.last_dof_vel = np.zeros(self.num_joints)
        self.command_steps = 0

        # 每回合随机化的摩擦系数（评估时不随机化）
        self.foot_friction = np.full(4, 1.0)

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.command_steps = COMMAND_RESAMPLE_STEPS  # 强制立即重采样

        # 初始位姿（训练时带小扰动，评估时从精确默认姿态开始）
        self.data.qpos[0] = float(self.rng.uniform(-0.5, 0.5))
        self.data.qpos[1] = float(self.rng.uniform(-0.5, 0.5))
        self.data.qpos[2] = BASE_HEIGHT_TARGET + 0.01 + float(self.rng.uniform(-0.005, 0.005))
        rpy = self.rng.uniform(-0.15, 0.15, 3) if self.randomize else np.zeros(3)
        self.data.qpos[3:7] = self._rpy_to_quat(rpy)
        dof_noise = self.rng.uniform(-0.05, 0.05, 12) if self.randomize else np.zeros(12)
        self.data.qpos[self.joint_qpos_adr:] = DEFAULT_DOF_POS + dof_noise
        self.data.qvel[:] = 0.0

        self.actions[:] = 0.0
        self.last_action[:] = 0.0
        self.last_dof_vel[:] = 0.0

        if self.randomize:
            self._randomize_friction()
        self._resample_commands(first=True)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.last_action = self.actions.copy()
        self.actions = action

        # 目标关节角 = 默认姿态 + 缩放后的动作
        target = DEFAULT_DOF_POS + ACTION_SCALE * action

        # 随机推一把（指令重采样时 10% 概率，模拟外力干扰）
        if self.randomize and self.command_steps == 0 and self.rng.random() < 0.1:
            self._apply_push()

        self.last_dof_vel = self.data.qvel[self.joint_qvel_adr:].copy()
        for _ in range(self.decimation):
            q = self.data.qpos[self.joint_qpos_adr:]
            dq = self.data.qvel[self.joint_qvel_adr:]
            tau = KP * (target - q) - KD * dq
            self.data.ctrl[:] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.command_steps += 1
        if self.command_steps >= COMMAND_RESAMPLE_STEPS:
            self._resample_commands(first=False)

        terminated = self._check_termination()
        truncated = self.step_count >= self.episode_length
        reward, reward_components = self._compute_reward(terminated)
        obs = self._get_obs()

        info = {
            "reward_components": reward_components,
            "commands": self.commands.copy(),
            "base_pos": self.data.qpos[0:3].copy(),
            "base_rpy": _quat_to_rpy(self.data.qpos[3:7]),
        }
        return obs, reward, terminated, truncated, info

    def set_commands(self, vx: float, vy: float, yaw_rate: float):
        """评估时由外部脚本设定指令（command_override=True 时生效）。"""
        self.commands[:] = [vx, vy, yaw_rate]

    def get_default_dof_pos(self) -> np.ndarray:
        return DEFAULT_DOF_POS.copy()

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[3:7]  # 基座四元数 [w,x,y,z]

        base_ang_vel_world = self.data.qvel[3:6]
        base_ang_vel = _quat_rotate_inverse(q, base_ang_vel_world)      # 转到机体系
        projected_gravity = _quat_rotate_inverse(q, np.array([0, 0, -1]))
        dof_pos = self.data.qpos[self.joint_qpos_adr:] - DEFAULT_DOF_POS
        dof_vel = self.data.qvel[self.joint_qvel_adr:]

        obs = np.concatenate([
            base_ang_vel,
            projected_gravity,
            self.commands,
            dof_pos,
            dof_vel,
            self.actions,
        ]) * self.obs_scales

        if self.add_noise:
            noise = np.concatenate([
                self.rng.normal(0.0, 0.2, 3),    # 角速度
                self.rng.normal(0.0, 0.05, 3),   # 投影重力
                np.zeros(3),                     # 指令不加噪
                self.rng.normal(0.0, 0.01, 12),  # 关节角
                self.rng.normal(0.0, 1.5, 12),   # 关节角速度
                np.zeros(12),                    # 动作不加噪
            ])
            obs += noise
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # 奖励
    # ------------------------------------------------------------------ #
    def _compute_reward(self, terminated: bool):
        q = self.data.qpos[3:7]
        base_lin_vel_world = self.data.qvel[0:3]
        base_lin_vel = _quat_rotate_inverse(q, base_lin_vel_world)   # 机体系速度
        base_ang_vel = _quat_rotate_inverse(q, self.data.qvel[3:6])
        projected_gravity = _quat_rotate_inverse(q, np.array([0, 0, -1]))
        dof_vel = self.data.qvel[self.joint_qvel_adr:]
        dof_acc = (dof_vel - self.last_dof_vel) / self.dt
        torques = self.data.ctrl.copy()
        base_height = self.data.qpos[2]

        cmd_vx, cmd_vy, cmd_yaw = self.commands

        r_lin_vel = float(np.exp(-((cmd_vx - base_lin_vel[0]) ** 2 +
                                   (cmd_vy - base_lin_vel[1]) ** 2) / 0.25))
        r_ang_vel = float(np.exp(-(cmd_yaw - base_ang_vel[2]) ** 2 / 0.25))
        r_height = float(np.exp(-(base_height - BASE_HEIGHT_TARGET) ** 2 * 10.0))

        # 关节限位：超出 [下限+10%量程, 上限-10%量程] 时惩罚
        margin = 0.1 * (DOF_UPPER - DOF_LOWER)
        q_joint = self.data.qpos[self.joint_qpos_adr:]
        violation = np.maximum(0.0, (DOF_LOWER + margin) - q_joint) + \
                    np.maximum(0.0, q_joint - (DOF_UPPER - margin))

        components = {
            "lin_vel": 1.0 * r_lin_vel,
            "ang_vel": 0.5 * r_ang_vel,
            "vel_z": -2.0 * float(base_lin_vel[2] ** 2),
            "ang_xy": -0.05 * float(base_ang_vel[0] ** 2 + base_ang_vel[1] ** 2),
            "height": 0.2 * r_height,
            "orient": -0.2 * float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2),
            "action_rate": -0.01 * float(np.sum((self.actions - self.last_action) ** 2)),
            "torque": -1e-5 * float(np.sum(torques ** 2)),
            "dof_acc": -2.5e-7 * float(np.sum(dof_acc ** 2)),
            "joint_limit": -10.0 * float(np.sum(violation ** 2)),
            "termination": -100.0 if terminated else 0.0,
        }
        return float(sum(components.values())), components

    # ------------------------------------------------------------------ #
    # 终止、指令、随机化
    # ------------------------------------------------------------------ #
    def _check_termination(self) -> bool:
        rpy = _quat_to_rpy(self.data.qpos[3:7])
        if abs(rpy[0]) > 0.5 or abs(rpy[1]) > 0.7:
            return True
        if self.data.qpos[2] < FALL_HEIGHT:
            return True
        # 躯干触地
        for c in self.data.contact:
            pair = {c.geom1, c.geom2}
            if self.torso_geom_id in pair and self.floor_geom_id in pair:
                return True
        return False

    def _resample_commands(self, first: bool):
        """每 10 s 重采样一次速度指令。回合开始时 50% 概率先给静止指令。"""
        self.command_steps = 0
        if self.command_override:
            return
        if first and self.rng.random() < 0.5:
            self.commands[:] = 0.0
        else:
            self.commands[:] = [
                float(self.rng.uniform(-1.0, 1.0)),
                float(self.rng.uniform(-0.6, 0.6)),
                float(self.rng.uniform(-1.0, 1.0)),
            ]

    def _randomize_friction(self):
        friction = float(self.rng.uniform(0.5, 1.25))
        self.foot_friction[:] = friction
        for gid in self.foot_geom_ids:
            self.model.geom_friction[gid, 0] = friction

    def _apply_push(self):
        """对基座施加一次随机水平速度脉冲。"""
        self.data.qvel[0] += float(self.rng.uniform(-1.0, 1.0))
        self.data.qvel[1] += float(self.rng.uniform(-1.0, 1.0))

    @staticmethod
    def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])
