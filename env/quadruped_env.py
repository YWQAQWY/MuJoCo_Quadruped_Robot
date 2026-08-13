"""四足机械狗 MuJoCo 环境（完整实现，无留空）。

接口风格对齐 gymnasium：
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

控制：策略输出 12 维动作（目标关节角偏移），PD 力矩计算在 control/pd_controller.py：
    tau = kp * (target - q) - kd * dq

所有超参数在 config/ 下：
    config/robot.yaml  默认站姿、关节限位、标称高度
    config/env.yaml    时序、指令、观测、奖励、终止、域随机化
    config/pd.yaml     PD 增益（control/pd_controller.py 读取）
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from config import load
from control.pd_controller import PDController

XML_PATH = Path(__file__).resolve().parent.parent / "robot" / "quadruped.xml"

ROBOT_CFG = load("robot")   # config/robot.yaml
ENV_CFG = load("env")       # config/env.yaml

# 导出常用量（供测试和脚本使用），内容来自 config/robot.yaml
DEFAULT_DOF_POS = np.array(ROBOT_CFG["default_dof_pos"], dtype=float)
DOF_LOWER = np.array(ROBOT_CFG["dof_lower"] * 4, dtype=float)
DOF_UPPER = np.array(ROBOT_CFG["dof_upper"] * 4, dtype=float)


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
        episode_length: int | None = None,   # 默认取 config/env.yaml
        decimation: int | None = None,       # 默认取 config/env.yaml
        add_noise: bool = True,              # 观测噪声（训练开、评估关）
        randomize: bool = True,              # 域随机化（训练开、评估关）
        command_override: bool = False,      # 评估时由外部脚本设定指令
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.cfg = ENV_CFG
        sim_cfg = self.cfg["simulation"]
        self.episode_length = int(sim_cfg["episode_length"] if episode_length is None else episode_length)
        self.decimation = int(sim_cfg["decimation"] if decimation is None else decimation)
        self.add_noise = add_noise
        self.randomize = randomize
        self.command_override = command_override
        self.dt = self.model.opt.timestep * self.decimation  # 控制周期 0.02 s

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
        # 关节数据在 qpos/qvel 中的起始地址：跳过躯干自由关节（jnt 0）
        self.joint_qpos_adr = self.model.jnt_qposadr[1]
        self.joint_qvel_adr = self.model.jnt_dofadr[1]

        # PD 控制器（control/pd_controller.py）：动作 → 目标关节角 → 力矩
        self.pd = PDController(
            self.data, DEFAULT_DOF_POS, self.joint_qpos_adr, self.joint_qvel_adr
        )

        # 观测缩放（config/env.yaml → observation.scales）
        osc = self.cfg["observation"]["scales"]
        self.obs_scales = np.concatenate([
            np.array([osc["base_ang_vel"]] * 3),        # base 角速度 (rad/s)
            np.array([osc["projected_gravity"]] * 3),   # 投影重力
            np.array(osc["commands"]),                  # 指令 vx, vy, yaw
            np.array([osc["dof_pos"]] * 12),            # 关节角 - 默认值
            np.array([osc["dof_vel"]] * 12),            # 关节角速度
            np.array([osc["actions"]] * 12),            # 上一步动作
        ])
        self.obs_noise_std = self.cfg["observation"]["noise"]

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
        self.command_steps = self.cfg["commands"]["resample_steps"]  # 强制立即重采样

        # 初始位姿（训练时带小扰动，评估时从精确默认姿态开始）
        init_cfg = self.cfg["domain_randomization"]["init"]
        pos_range = init_cfg["pos_range"]
        self.data.qpos[0] = float(self.rng.uniform(*pos_range))
        self.data.qpos[1] = float(self.rng.uniform(*pos_range))
        self.data.qpos[2] = float(ROBOT_CFG["base_height_target"] + 0.01
                                  + self.rng.uniform(-init_cfg["z_noise"], init_cfg["z_noise"]))
        rpy = self.rng.uniform(-init_cfg["rpy_noise"], init_cfg["rpy_noise"], 3) \
            if self.randomize else np.zeros(3)
        self.data.qpos[3:7] = self._rpy_to_quat(rpy)
        dof_noise = self.rng.uniform(-init_cfg["dof_pos_noise"], init_cfg["dof_pos_noise"], 12) \
            if self.randomize else np.zeros(12)
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

        # 动作 → 目标关节角（PD 控制器内部完成）
        self.pd.set_action(action)

        # 随机推一把（指令重采样时按概率触发，模拟外力干扰）
        push_cfg = self.cfg["domain_randomization"]["push"]
        if self.randomize and self.command_steps == 0 and self.rng.random() < push_cfg["probability"]:
            self._apply_push()

        self.last_dof_vel = self.data.qvel[self.joint_qvel_adr:].copy()
        for _ in range(self.decimation):
            self.data.ctrl[:] = self.pd.compute_torques()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.command_steps += 1
        if self.command_steps >= self.cfg["commands"]["resample_steps"]:
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
            n = self.obs_noise_std
            noise = np.concatenate([
                self.rng.normal(0.0, n["base_ang_vel"], 3),
                self.rng.normal(0.0, n["projected_gravity"], 3),
                np.zeros(3),                        # 指令不加噪
                self.rng.normal(0.0, n["dof_pos"], 12),
                self.rng.normal(0.0, n["dof_vel"], 12),
                np.zeros(12),                       # 动作不加噪
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
        r_cfg = self.cfg["rewards"]
        sigma = r_cfg["tracking_sigma"]
        base_height_target = ROBOT_CFG["base_height_target"]

        r_lin_vel = float(np.exp(-((cmd_vx - base_lin_vel[0]) ** 2 +
                                   (cmd_vy - base_lin_vel[1]) ** 2) / sigma))
        r_ang_vel = float(np.exp(-(cmd_yaw - base_ang_vel[2]) ** 2 / sigma))
        r_height = float(np.exp(-(base_height - base_height_target) ** 2
                                * r_cfg["base_height_sigma"]))

        # 关节限位：超出 [下限+margin·量程, 上限−margin·量程] 时惩罚
        margin = r_cfg["joint_limit_margin"] * (DOF_UPPER - DOF_LOWER)
        q_joint = self.data.qpos[self.joint_qpos_adr:]
        violation = np.maximum(0.0, (DOF_LOWER + margin) - q_joint) + \
                    np.maximum(0.0, q_joint - (DOF_UPPER - margin))

        w = r_cfg["weights"]
        components = {
            "lin_vel": w["lin_vel"] * r_lin_vel,
            "ang_vel": w["ang_vel"] * r_ang_vel,
            "vel_z": w["vel_z"] * float(base_lin_vel[2] ** 2),
            "ang_xy": w["ang_xy"] * float(base_ang_vel[0] ** 2 + base_ang_vel[1] ** 2),
            "height": w["base_height"] * r_height,
            "orient": w["orientation"] * float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2),
            "action_rate": w["action_rate"] * float(np.sum((self.actions - self.last_action) ** 2)),
            "torque": w["torque"] * float(np.sum(torques ** 2)),
            "dof_acc": w["dof_acc"] * float(np.sum(dof_acc ** 2)),
            "joint_limit": w["joint_limit"] * float(np.sum(violation ** 2)),
            "termination": w["termination"] if terminated else 0.0,
        }
        return float(sum(components.values())), components

    # ------------------------------------------------------------------ #
    # 终止、指令、随机化
    # ------------------------------------------------------------------ #
    def _check_termination(self) -> bool:
        term_cfg = self.cfg["termination"]
        rpy = _quat_to_rpy(self.data.qpos[3:7])
        if abs(rpy[0]) > term_cfg["max_roll"] or abs(rpy[1]) > term_cfg["max_pitch"]:
            return True
        if self.data.qpos[2] < ROBOT_CFG["fall_height"]:
            return True
        # 躯干触地
        if term_cfg.get("torso_contact", True):
            for c in self.data.contact:
                pair = {c.geom1, c.geom2}
                if self.torso_geom_id in pair and self.floor_geom_id in pair:
                    return True
        return False

    def _resample_commands(self, first: bool):
        """按 config/env.yaml 的周期和范围重采样速度指令。"""
        cmd_cfg = self.cfg["commands"]
        self.command_steps = 0
        if self.command_override:
            return
        if first and self.rng.random() < cmd_cfg["stand_still_prob"]:
            self.commands[:] = 0.0
        else:
            r = cmd_cfg["ranges"]
            self.commands[:] = [
                float(self.rng.uniform(*r["vx"])),
                float(self.rng.uniform(*r["vy"])),
                float(self.rng.uniform(*r["yaw_rate"])),
            ]

    def _randomize_friction(self):
        lo, hi = self.cfg["domain_randomization"]["friction_range"]
        friction = float(self.rng.uniform(lo, hi))
        self.foot_friction[:] = friction
        for gid in self.foot_geom_ids:
            self.model.geom_friction[gid, 0] = friction

    def _apply_push(self):
        """对基座施加一次随机水平速度脉冲。"""
        lo, hi = self.cfg["domain_randomization"]["push"]["velocity_range"]
        self.data.qvel[0] += float(self.rng.uniform(lo, hi))
        self.data.qvel[1] += float(self.rng.uniform(lo, hi))

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
