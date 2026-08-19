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

from config import PROJECT_DIR, deep_merge, load
from control.pd_controller import PDController

ROBOT_CFG = load("robot")   # config/robot.yaml
ENV_CFG = load("env")       # config/env.yaml
XML_PATH = PROJECT_DIR / ROBOT_CFG["xml_path"]

# 导出常用量（供测试和脚本使用），内容来自 config/robot.yaml
DEFAULT_DOF_POS = np.array(ROBOT_CFG["default_dof_pos"], dtype=float)
DOF_LOWER = np.tile(ROBOT_CFG["dof_lower"], len(ROBOT_CFG["leg_names"])).astype(float)
DOF_UPPER = np.tile(ROBOT_CFG["dof_upper"], len(ROBOT_CFG["leg_names"])).astype(float)


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
        include_base_lin_vel: bool = True,   # 新策略为 48 维；旧 checkpoint 可使用 45 维
        include_gait_obs: bool = True,       # phase(2)+足端接触(4)，新版共 54 维
        include_heading_obs: bool = True,    # 航向误差 sin/cos + 低通偏航率(3)，新版共 57 维
        config_override: dict | None = None, # 分阶段训练只覆盖需要改变的 env 字段
        pd_config_override: dict | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.cfg = deep_merge(ENV_CFG, config_override)
        sim_cfg = self.cfg["simulation"]
        self.episode_length = int(sim_cfg["episode_length"] if episode_length is None else episode_length)
        self.decimation = int(sim_cfg["decimation"] if decimation is None else decimation)
        self.add_noise = add_noise
        self.randomize = randomize
        self.command_override = command_override
        self.include_base_lin_vel = include_base_lin_vel
        self.include_gait_obs = include_gait_obs
        self.include_heading_obs = include_heading_obs
        self.dt = self.model.opt.timestep * self.decimation  # 控制周期 0.02 s

        self.num_joints = len(DEFAULT_DOF_POS)
        self.act_dim = self.num_joints
        self.obs_dim = ((3 if include_base_lin_vel else 0) + 3 + 3 + 3 +
                        3 * self.num_joints + (6 if include_gait_obs else 0) +
                        (3 if include_heading_obs else 0))
        self.step_count = 0

        # 运动学/动力学索引
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_CFG["torso_body_name"])
        self.torso_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_CFG["torso_geom_name"])
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, ROBOT_CFG["floor_geom_name"])
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"{leg}{ROBOT_CFG['foot_geom_suffix']}")
            for leg in ROBOT_CFG["leg_names"]
        ]
        self.foot_body_ids = [int(self.model.geom_bodyid[gid]) for gid in self.foot_geom_ids]
        self.foot_geom_id_to_index = {gid: i for i, gid in enumerate(self.foot_geom_ids)}
        self.foot_body_id_to_index = {bid: i for i, bid in enumerate(self.foot_body_ids)}
        self._base_body_mass = self.model.body_mass.copy()
        self._base_body_inertia = self.model.body_inertia.copy()
        self._base_kp = self.pd.kp if hasattr(self, "pd") else None
        # 关节数据在 qpos/qvel 中的起始地址：跳过躯干自由关节（jnt 0）
        self.joint_qpos_adr = self.model.jnt_qposadr[1]
        self.joint_qvel_adr = self.model.jnt_dofadr[1]

        # PD 控制器（control/pd_controller.py）：动作 → 目标关节角 → 力矩
        self.pd = PDController(
            self.data, DEFAULT_DOF_POS, self.joint_qpos_adr, self.joint_qvel_adr,
            config_override=pd_config_override,
        )
        self._base_kp = self.pd.kp
        self._base_kd = self.pd.kd

        # 观测缩放（config/env.yaml → observation.scales）
        osc = self.cfg["observation"]["scales"]
        scale_parts = []
        if self.include_base_lin_vel:
            scale_parts.append(np.array([osc["base_lin_vel"]] * 3))
        scale_parts.extend([
            np.array([osc["base_ang_vel"]] * 3),        # base 角速度 (rad/s)
            np.array([osc["projected_gravity"]] * 3),   # 投影重力
            np.array(osc["commands"]),                  # 指令 vx, vy, yaw
            np.full(self.num_joints, osc["dof_pos"]),   # 关节角 - 默认值
            np.full(self.num_joints, osc["dof_vel"]),   # 关节角速度
            np.full(self.num_joints, osc["actions"]),   # 上一步动作
        ])
        if self.include_gait_obs:
            scale_parts.extend([
                np.full(2, osc["phase"]),
                np.full(len(self.foot_geom_ids), osc["foot_contacts"]),
            ])
        if self.include_heading_obs:
            scale_parts.append(np.array([osc["heading"], osc["heading"], osc["yaw_ema"]]))
        self.obs_scales = np.concatenate(scale_parts)
        self.obs_noise_std = self.cfg["observation"]["noise"]

        self.commands = np.zeros(3)         # [vx, vy, yaw_rate]，机体坐标系
        self.actions = np.zeros(self.act_dim)
        self.last_action = np.zeros(self.act_dim)
        self.last_dof_vel = np.zeros(self.num_joints)
        self.command_steps = 0
        self.gait_phase = 0.0
        self.foot_air_time = np.zeros(len(self.foot_geom_ids))
        self.last_foot_contacts = np.zeros(len(self.foot_geom_ids), dtype=bool)
        self.feet_air_time_reward = 0.0
        self.yaw_ema = 0.0
        self.curriculum_command_scale = 1.0
        self.curriculum_randomization_scale = 1.0
        self.curriculum_residual_scale = float(self.cfg["gait"]["residual_control"]["scale"])
        self.curriculum_frequency_scale = 1.0
        self.action_delay = 0
        self.action_buffer = []
        gait_cfg = self.cfg["gait"]
        self.gait_phase = (float(self.rng.random())
                           if self.randomize and gait_cfg["randomize_initial_phase"] else 0.0)
        self.foot_air_time[:] = 0.0
        self.last_foot_contacts[:] = False
        self.feet_air_time_reward = 0.0

        # 每回合随机化的摩擦系数（评估时不随机化）
        self.foot_friction = np.full(len(self.foot_geom_ids), 1.0)
        self.foot_contact_forces = np.zeros(len(self.foot_geom_ids))
        self.body_contact_count = 0
        self.torso_contact_count = 0
        self.reference_action = np.zeros(self.act_dim)
        self.initial_yaw = 0.0

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.command_steps = self.cfg["commands"]["resample_steps"]  # 强制立即重采样

        # 初始位姿（训练时带小扰动，评估时从精确默认姿态开始）
        init_cfg = self.cfg["domain_randomization"]["init"]
        randomization_scale = self.curriculum_randomization_scale if self.randomize else 0.0
        pos_range = init_cfg["pos_range"]
        self.data.qpos[0] = float(self.rng.uniform(*pos_range))
        self.data.qpos[1] = float(self.rng.uniform(*pos_range))
        self.data.qpos[2] = float(ROBOT_CFG["base_height_target"] + init_cfg["base_height_offset"]
                                  + self.rng.uniform(-init_cfg["z_noise"], init_cfg["z_noise"])
                                  * randomization_scale)
        rpy_noise = init_cfg["rpy_noise"] * randomization_scale
        rpy = self.rng.uniform(-rpy_noise, rpy_noise, 3) \
            if self.randomize else np.zeros(3)
        self.data.qpos[3:7] = self._rpy_to_quat(rpy)
        dof_pos_noise = init_cfg["dof_pos_noise"] * randomization_scale
        dof_noise = self.rng.uniform(-dof_pos_noise, dof_pos_noise,
                                     self.num_joints) \
            if self.randomize else np.zeros(self.num_joints)
        self.data.qpos[self.joint_qpos_adr:] = DEFAULT_DOF_POS + dof_noise
        self.data.qvel[:] = 0.0

        self.actions[:] = 0.0
        self.last_action[:] = 0.0
        self.last_dof_vel[:] = 0.0
        self.pd.target = DEFAULT_DOF_POS.copy()
        self.action_buffer = []
        gait_cfg = self.cfg["gait"]
        self.gait_phase = (float(self.rng.random())
                           if self.randomize and gait_cfg["randomize_initial_phase"] else 0.0)
        self.foot_air_time[:] = 0.0
        self.last_foot_contacts[:] = False
        self.feet_air_time_reward = 0.0
        self.yaw_ema = 0.0

        if self.randomize:
            self._randomize_friction()
            self._randomize_dynamics()
        else:
            self._restore_dynamics()
        self._resample_commands(first=True)

        mujoco.mj_forward(self.model, self.data)
        self.initial_yaw = float(_quat_to_rpy(self.data.qpos[3:7])[2])
        self.last_foot_contacts = self._get_foot_contacts()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.action_buffer.append(action.copy())
        if len(self.action_buffer) <= self.action_delay:
            action = np.zeros_like(action)
        else:
            action = self.action_buffer.pop(0)
        self.last_action = self.actions.copy()
        self.actions = action

        # 残差控制：默认站姿 + 参数化参考小跑 + PPO 修正量。
        self.reference_action = self._get_reference_joint_offsets()
        residual_cfg = self.cfg["gait"]["residual_control"]
        residual_offsets = (self.pd.action_scale * self.curriculum_residual_scale * action
                            if residual_cfg["enabled"] else self.pd.action_scale * action)
        self.pd.set_target_offsets(self.reference_action + residual_offsets)

        # 随机推一把（指令重采样时按概率触发，模拟外力干扰）
        push_cfg = self.cfg["domain_randomization"]["push"]
        push_probability = push_cfg["probability"] * self.curriculum_randomization_scale
        if self.randomize and self.command_steps == 0 and self.rng.random() < push_probability:
            self._apply_push()

        self.last_dof_vel = self.data.qvel[self.joint_qvel_adr:].copy()
        for _ in range(self.decimation):
            self.data.ctrl[:] = self.pd.compute_torques()
            mujoco.mj_step(self.model, self.data)

        frequency = self.cfg["gait"]["phase_frequency"] * self.curriculum_frequency_scale
        self.gait_phase = (self.gait_phase + self.dt * frequency) % 1.0
        self._update_foot_air_time()

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
            "base_lin_vel": _quat_rotate_inverse(
                self.data.qpos[3:7], self.data.qvel[0:3]).copy(),
            "base_ang_vel": _quat_rotate_inverse(
                self.data.qpos[3:7], self.data.qvel[3:6]).copy(),
            "foot_contacts": self.last_foot_contacts.copy(),
            "foot_contact_forces": self.foot_contact_forces.copy(),
            "foot_heights": self._get_foot_heights(),
            "reference_action": self.reference_action.copy(),
            "body_contact_count": self.body_contact_count,
            "torso_contact_count": self.torso_contact_count,
        }
        return obs, reward, terminated, truncated, info

    def set_commands(self, vx: float, vy: float, yaw_rate: float):
        """评估时由外部脚本设定指令（command_override=True 时生效）。"""
        self.commands[:] = [vx, vy, yaw_rate]

    def set_curriculum(self, command_scale: float, randomization_scale: float,
                       residual_scale: float | None = None,
                       frequency_scale: float | None = None):
        """设置课程难度；范围均为 [0, 1]，通常由训练循环逐步提升。"""
        self.curriculum_command_scale = float(np.clip(command_scale, 0.0, 1.0))
        self.curriculum_randomization_scale = float(np.clip(randomization_scale, 0.0, 1.0))
        if residual_scale is not None:
            self.curriculum_residual_scale = float(np.clip(residual_scale, 0.0, 1.0))
        if frequency_scale is not None:
            self.curriculum_frequency_scale = float(max(0.0, frequency_scale))

    def get_default_dof_pos(self) -> np.ndarray:
        return DEFAULT_DOF_POS.copy()

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[3:7]  # 基座四元数 [w,x,y,z]

        base_lin_vel = _quat_rotate_inverse(q, self.data.qvel[0:3])
        base_ang_vel_world = self.data.qvel[3:6]
        base_ang_vel = _quat_rotate_inverse(q, base_ang_vel_world)      # 转到机体系
        projected_gravity = _quat_rotate_inverse(q, np.array([0, 0, -1]))
        dof_pos = self.data.qpos[self.joint_qpos_adr:] - DEFAULT_DOF_POS
        dof_vel = self.data.qvel[self.joint_qvel_adr:]

        obs_parts = []
        if self.include_base_lin_vel:
            obs_parts.append(base_lin_vel)
        obs_parts.extend([
            base_ang_vel,
            projected_gravity,
            self.commands,
            dof_pos,
            dof_vel,
            self.actions,
        ])
        if self.include_gait_obs:
            angle = 2.0 * np.pi * self.gait_phase
            obs_parts.extend([
                np.array([np.sin(angle), np.cos(angle)]),
                self._get_foot_contacts().astype(float),
            ])
        if self.include_heading_obs:
            # 航向误差进入观测：straight_heading 惩罚依赖它，actor/critic 必须能直接看到
            current_yaw = float(_quat_to_rpy(q)[2])
            heading = float(np.arctan2(np.sin(current_yaw - self.initial_yaw),
                                       np.cos(current_yaw - self.initial_yaw)))
            obs_parts.append(np.array([np.sin(heading), np.cos(heading), self.yaw_ema]))
        obs = np.concatenate(obs_parts) * self.obs_scales

        if self.add_noise:
            n = self.obs_noise_std
            noise_parts = []
            if self.include_base_lin_vel:
                noise_parts.append(self.rng.normal(0.0, n["base_lin_vel"], 3))
            noise_parts.extend([
                self.rng.normal(0.0, n["base_ang_vel"], 3),
                self.rng.normal(0.0, n["projected_gravity"], 3),
                np.zeros(3),                        # 指令不加噪
                self.rng.normal(0.0, n["dof_pos"], self.num_joints),
                self.rng.normal(0.0, n["dof_vel"], self.num_joints),
                np.zeros(self.num_joints),          # 动作不加噪
            ])
            if self.include_gait_obs:
                noise_parts.extend([np.zeros(2), np.zeros(len(self.foot_geom_ids))])
            if self.include_heading_obs:
                noise_parts.append(np.zeros(3))
            noise = np.concatenate(noise_parts)
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
        current_yaw = float(_quat_to_rpy(q)[2])
        heading_error = float(np.arctan2(np.sin(current_yaw - self.initial_yaw),
                                         np.cos(current_yaw - self.initial_yaw)))

        cmd_vx, cmd_vy, cmd_yaw = self.commands
        r_cfg = self.cfg["rewards"]
        # 低通滤波偏航率：隔离持续漂移与正常步态振荡，惩罚即时且可观测
        self.yaw_ema = ((1.0 - float(r_cfg["yaw_drift_alpha"])) * self.yaw_ema +
                        float(r_cfg["yaw_drift_alpha"]) * float(base_ang_vel[2]))
        base_height_target = ROBOT_CFG["base_height_target"]

        r_lin_vel = float(np.exp(-((cmd_vx - base_lin_vel[0]) ** 2 +
                                   (cmd_vy - base_lin_vel[1]) ** 2) /
                                 r_cfg["lin_tracking_sigma"]))
        r_ang_vel = float(np.exp(-(cmd_yaw - base_ang_vel[2]) ** 2 /
                                 r_cfg["ang_tracking_sigma"]))
        r_height = float(np.exp(-(base_height - base_height_target) ** 2
                                * r_cfg["base_height_sigma"]))

        # 关节限位：超出 [下限+margin·量程, 上限−margin·量程] 时惩罚
        margin = r_cfg["joint_limit_margin"] * (DOF_UPPER - DOF_LOWER)
        q_joint = self.data.qpos[self.joint_qpos_adr:]
        violation = np.maximum(0.0, (DOF_LOWER + margin) - q_joint) + \
                    np.maximum(0.0, q_joint - (DOF_UPPER - margin))

        contact_data = self._contact_state()
        foot_contacts = contact_data["contact_geoms"]
        undesired_contacts = contact_data["body_contacts"]
        feet_slip = 0.0
        for geom_id in foot_contacts:
            body_id = int(self.model.geom_bodyid[geom_id])
            feet_slip += float(np.sum(self.data.cvel[body_id, 3:5] ** 2))
        command_norm = float(np.linalg.norm(self.commands))
        linear_command_norm = float(np.linalg.norm(self.commands[:2]))
        horizontal_speed = float(np.linalg.norm(base_lin_vel[:2]))
        no_motion = (linear_command_norm > r_cfg["moving_command_threshold"] and
                     horizontal_speed < r_cfg["stationary_velocity_threshold"])
        # Symmetric forward speed error: overshoot is penalised as much as
        # undershoot, so a uniform speed bias cannot pay off.
        forward_speed_error = abs(float(cmd_vx) - float(base_lin_vel[0]))
        foot_contacts_bool = contact_data["contacts"]
        gait_terms = self._gait_style_terms(foot_contacts_bool, contact_data["forces"])
        moving = command_norm > r_cfg["moving_command_threshold"]

        w = r_cfg["weights"]
        components = {
            "lin_vel": w["lin_vel"] * r_lin_vel,
            "ang_vel": w["ang_vel"] * r_ang_vel,
            "vel_z": w["vel_z"] * float(base_lin_vel[2] ** 2),
            "ang_xy": w["ang_xy"] * float(base_ang_vel[0] ** 2 + base_ang_vel[1] ** 2),
            "lateral_velocity": w["lateral_velocity"] * float(base_lin_vel[1] ** 2),
            "straight_heading": w["straight_heading"] * heading_error ** 2
                                if moving and abs(cmd_yaw) < r_cfg["straight_yaw_threshold"]
                                else 0.0,
            "yaw_drift": w["yaw_drift"] * float(abs(self.yaw_ema))
                         if moving and abs(cmd_yaw) < r_cfg["straight_yaw_threshold"]
                         else 0.0,
            "height": w["base_height"] * r_height,
            "orient": w["orientation"] * float(projected_gravity[0] ** 2 + projected_gravity[1] ** 2),
            "action_rate": w["action_rate"] * float(np.sum((self.actions - self.last_action) ** 2)),
            "torque": w["torque"] * float(np.sum(torques ** 2)),
            "dof_acc": w["dof_acc"] * float(np.sum(dof_acc ** 2)),
            "joint_limit": w["joint_limit"] * float(np.sum(violation ** 2)),
            "power": w["power"] * float(np.sum(np.abs(torques * dof_vel))),
            "feet_slip": w["feet_slip"] * feet_slip,
            "undesired_contact": w["undesired_contact"] * undesired_contacts,
            "stand_pose": w["stand_pose"] * float(np.sum((q_joint - DEFAULT_DOF_POS) ** 2))
                          if command_norm < r_cfg["stand_command_threshold"] else 0.0,
            "no_motion": w["no_motion"] if no_motion else 0.0,
            "speed_error": w["speed_error"] * forward_speed_error
                           if moving else 0.0,
            "gait_contact": w["gait_contact"] * gait_terms["contact"] if moving else 0.0,
            "foot_clearance": w["foot_clearance"] * gait_terms["clearance"] if moving else 0.0,
            "landing_velocity": w["landing_velocity"] * gait_terms["landing_velocity"]
                                if moving else 0.0,
            "feet_air_time": w["feet_air_time"] * self.feet_air_time_reward if moving else 0.0,
            "termination": w["termination"] if terminated else 0.0,
        }
        return float(sum(components.values())), components

    def _get_foot_contacts(self) -> np.ndarray:
        """Return force-thresholded contacts in FL/FR/RL/RR order."""
        return self._contact_state()["contacts"]

    def _contact_state(self) -> dict:
        """Classify floor contacts and aggregate MuJoCo normal forces per foot."""
        forces = np.zeros(len(self.foot_body_ids))
        contact_geoms, body_contacts, torso_contacts = set(), 0, 0
        wrench = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            pair = {contact.geom1, contact.geom2}
            if self.floor_geom_id not in pair:
                continue
            other = contact.geom2 if contact.geom1 == self.floor_geom_id else contact.geom1
            body_id = int(self.model.geom_bodyid[other])
            foot_index = self.foot_body_id_to_index.get(body_id)
            if foot_index is not None:
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                forces[foot_index] += abs(float(wrench[0]))
                contact_geoms.add(other)
            elif other == self.torso_geom_id:
                torso_contacts += 1
            else:
                body_contacts += 1
        gait_cfg = self.cfg["gait"]
        on_threshold = float(gait_cfg["contact_force_threshold"])
        off_threshold = float(gait_cfg["contact_force_release_threshold"])
        contacts = np.where(self.last_foot_contacts,
                            forces >= off_threshold, forces >= on_threshold)
        self.foot_contact_forces = forces
        self.body_contact_count = body_contacts
        self.torso_contact_count = torso_contacts
        return {"contacts": contacts.astype(bool), "forces": forces,
                "contact_geoms": contact_geoms, "body_contacts": body_contacts,
                "torso_contacts": torso_contacts}

    def _phase_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Smooth diagonal-trot stance/swing weights for FL/FR/RL/RR."""
        cfg = self.cfg["gait"]
        phase_offsets = np.asarray(cfg["leg_phase_offsets"], dtype=float)
        local = (self.gait_phase + phase_offsets) % 1.0
        duty = float(cfg["duty_factor"])
        smoothing = max(float(cfg["phase_smoothing"]), 1e-4)
        # Smooth periodic interval [0, duty] using circular distance to its centre.
        distance = np.abs((local - duty / 2.0 + 0.5) % 1.0 - 0.5)
        stance = 1.0 / (1.0 + np.exp((distance - duty / 2.0) / smoothing))
        return stance, 1.0 - stance

    def _get_foot_heights(self) -> np.ndarray:
        return np.asarray([self.data.geom_xpos[gid, 2] for gid in self.foot_geom_ids])

    def _get_foot_vertical_velocities(self) -> np.ndarray:
        result = np.zeros(len(self.foot_body_ids))
        velocity = np.zeros(6)
        for i, body_id in enumerate(self.foot_body_ids):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                     body_id, velocity, 0)
            result[i] = velocity[5]
        return result

    def _gait_style_terms(self, contacts: np.ndarray, forces: np.ndarray) -> dict:
        """Zero-at-best style costs; all-feet-down always receives a contact penalty."""
        stance, swing = self._phase_weights()
        cfg = self.cfg["gait"]
        force_scale = max(float(cfg["contact_force_normalization"]), 1e-6)
        contact_strength = np.clip(forces / force_scale, 0.0, 1.0)
        swing_contact = np.mean(swing * contact_strength)
        stance_miss = np.mean(stance * (~contacts).astype(float))
        heights = self._get_foot_heights()
        clearance_error = np.mean(swing * np.maximum(
            float(cfg["swing_foot_height"]) - heights, 0.0) ** 2)
        vertical_velocity = self._get_foot_vertical_velocities()
        touchdown = contacts & ~self.last_foot_contacts
        landing_cost = np.mean(touchdown * np.maximum(-vertical_velocity, 0.0) ** 2)
        return {"contact": -float(swing_contact + stance_miss),
                "clearance": -float(clearance_error),
                "landing_velocity": -float(landing_cost)}

    def _get_reference_joint_offsets(self) -> np.ndarray:
        """Parameterized diagonal-trot reference in radians, disabled near zero command."""
        cfg = self.cfg["gait"]["reference"]
        moving = float(np.linalg.norm(self.commands)) > self.cfg["rewards"]["moving_command_threshold"]
        if not cfg["enabled"] or not moving:
            return np.zeros(self.act_dim)
        speed = float(np.linalg.norm(self.commands[:2]))
        speed_scale = np.clip(speed / max(float(cfg["nominal_speed"]), 1e-6),
                              float(cfg["minimum_speed_scale"]),
                              float(cfg["maximum_speed_scale"]))
        lift_scale = np.clip(speed / max(float(cfg["nominal_speed"]), 1e-6),
                             float(cfg["minimum_lift_scale"]),
                             float(cfg["maximum_lift_scale"]))
        phase_offsets = np.asarray(self.cfg["gait"]["leg_phase_offsets"], dtype=float)
        local = (self.gait_phase + phase_offsets) % 1.0
        stance, swing = self._phase_weights()
        del stance
        offsets = np.zeros((len(phase_offsets), 3))
        amplitudes = np.asarray(cfg["joint_amplitudes"], dtype=float)
        biases = np.asarray(cfg["joint_biases"], dtype=float)
        wave = np.sin(2.0 * np.pi * local + float(cfg["phase_bias"]))
        offsets[:] = biases + speed_scale * wave[:, None] * amplitudes[None, :]
        # Leg clearance must not collapse at low speed. Stride length follows the
        # command, while swing knee lift has an independent lower bound.
        offsets[:, 2] += lift_scale * float(cfg["swing_knee_lift"]) * swing
        offsets[:, 0] *= np.asarray(cfg["abduction_leg_signs"], dtype=float)
        return offsets.reshape(-1)

    def _update_foot_air_time(self):
        contacts = self._get_foot_contacts()
        self.foot_air_time += self.dt
        touchdown = contacts & ~self.last_foot_contacts
        target = self.cfg["gait"]["air_time_target"]
        self.feet_air_time_reward = float(np.sum(np.maximum(self.foot_air_time - target, 0.0)
                                                 * touchdown))
        self.foot_air_time[contacts] = 0.0
        self.last_foot_contacts = contacts

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
        progress = self.curriculum_command_scale
        stand_prob = cmd_cfg["stand_still_prob"] + progress * (
            cmd_cfg["stand_still_prob_final"] - cmd_cfg["stand_still_prob"]
        )
        if self.rng.random() < stand_prob:
            self.commands[:] = 0.0
        else:
            start_ranges = cmd_cfg["curriculum_start_ranges"]
            final_ranges = cmd_cfg["ranges"]
            r = {}
            for key in ("vx", "vy", "yaw_rate"):
                start = np.asarray(start_ranges[key], dtype=float)
                final = np.asarray(final_ranges[key], dtype=float)
                r[key] = start + progress * (final - start)
            self.commands[:] = [
                float(self.rng.uniform(*r["vx"])),
                float(self.rng.uniform(*r["vy"])),
                float(self.rng.uniform(*r["yaw_rate"])),
            ]

    def _randomize_friction(self):
        scale = self.curriculum_randomization_scale
        lo, hi = self.cfg["domain_randomization"]["friction_range"]
        lo, hi = 1.0 + (lo - 1.0) * scale, 1.0 + (hi - 1.0) * scale
        friction = float(self.rng.uniform(lo, hi))
        self.foot_friction[:] = friction
        for gid in self.foot_geom_ids:
            self.model.geom_friction[gid, 0] = friction

    def _restore_dynamics(self):
        self.model.body_mass[:] = self._base_body_mass
        self.model.body_inertia[:] = self._base_body_inertia
        self.pd.kp = self._base_kp
        self.pd.kd = self._base_kd
        self.action_delay = 0

    def _randomize_dynamics(self):
        cfg = self.cfg["domain_randomization"]
        scale = self.curriculum_randomization_scale
        mass_lo, mass_hi = cfg["mass_scale_range"]
        mass_range = (1.0 + (mass_lo - 1.0) * scale, 1.0 + (mass_hi - 1.0) * scale)
        mass_scale = float(self.rng.uniform(*mass_range))
        self.model.body_mass[:] = self._base_body_mass * mass_scale
        self.model.body_inertia[:] = self._base_body_inertia * mass_scale
        gain_lo, gain_hi = cfg["pd_gain_scale_range"]
        gain_range = (1.0 + (gain_lo - 1.0) * scale, 1.0 + (gain_hi - 1.0) * scale)
        gain_scale = float(self.rng.uniform(*gain_range))
        self.pd.kp = self._base_kp * gain_scale
        self.pd.kd = self._base_kd * gain_scale
        lo, configured_hi = cfg["action_delay_steps"]
        hi = lo + int(round((configured_hi - lo) * scale))
        self.action_delay = int(self.rng.integers(lo, hi + 1))

    def _apply_push(self):
        """对基座施加一次随机水平速度脉冲。"""
        lo, hi = self.cfg["domain_randomization"]["push"]["velocity_range"]
        scale = self.curriculum_randomization_scale
        self.data.qvel[0] += float(self.rng.uniform(lo, hi)) * scale
        self.data.qvel[1] += float(self.rng.uniform(lo, hi)) * scale

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
