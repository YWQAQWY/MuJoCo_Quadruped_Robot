# MuJoCo Quadruped Robot

A 12-DOF quadruped robot modeled from scratch in MuJoCo, trained with a **from-scratch PPO
implementation** to walk and track velocity commands (forward vx / lateral vy / yaw rate).
No off-the-shelf RL libraries (stable-baselines3, etc.) are used.

## Algorithm Source

The PPO implementation is adapted from the RL algorithm tutorial repository
**[YWQAQWY/ReinforcementLearningAlgorithm](https://github.com/YWQAQWY/ReinforcementLearningAlgorithm.git)**:

- `PPO.py` (project root) started from the repository's **discrete-action PPO** (CartPole-v1, softmax + Categorical)
- It has been converted to a **tanh-squashed Gaussian policy** for the quadruped: the actor outputs
  mean μ and learnable log σ, while actions and log-probabilities use the same bounded transform.
  The entropy bonus estimates entropy after tanh instead of rewarding unbounded pre-tanh variance
- The repository's `PPO_continuous.py` (continuous PPO for Pendulum-v1) was the reference for this conversion

Core algorithm: PPO-Clip + GAE (Schulman et al. 2015/2017):

```
ratio = exp(logπ_new − logπ_old)
L_CLIP = mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))
loss = −L_CLIP + value_coef·0.5·mean((V−R)²) − entropy_coef·mean(entropy)
```

## PPO → PD Control Chain

How the trained PPO policy drives the robot, end to end:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  PPO policy (runs at 50 Hz in PPO.py)                                  │
  │                                                                        │
  │  obs (48-d) ──► PolicyNet (tanh Gaussian) ──► action a ∈ R¹²          │
  │                [base ang vel, projected  [clipped to [-1, 1]]          │
  │                 gravity, commands, joint                               │
  │                 pos/vel, previous action]                              │
  └───────────┬──────────────────────────────────────────┬─────────────────┘
              │ action a (12 joint-target offsets)       │ obs (next state)
              ▼                                          │
  ┌──────────────────────────────────────────────────────┴─────────────────┐
  │  env/quadruped_env.py + control/pd_controller.py (50 Hz)               │
  │                                                                        │
  │  1. target joint angles:  q_target = q_default + 0.25 · a              │
  │  2. PD torque per joint:  τ = 20·(q_target − q) − 0.5·q̇               │
  │  3. torque limit:         τ ← clip(τ, ±33.5 Nm)                        │
  └───────────┬────────────────────────────────────────────────────────────┘
              ▼  τ (12 motor torques)
  ┌────────────────────────────────────────────────────────────────────────┐
  │  MuJoCo physics (dt = 0.005 s, 200 Hz)                                 │
  │  4 simulation steps per control step ──► new state (q, q̇, base pose)   │
  └───────────┬────────────────────────────────────────────────────────────┘
              │
              ├──► reward (velocity tracking + penalties) ──► GAE ──► PPO update
              └──► next obs (48-d，含机体线速度) ──► back to policy
```

- **Policy outputs actions, not torques.** Each of the 12 actions is an offset added to the
  default joint pose, scaled by 0.25 rad — so the policy "thinks" in joint-angle space.
- **PD turns angles into torques.** The PD law lives in `control/pd_controller.py`
  (`set_action` + `compute_torques`); the env calls it every simulation step (200 Hz),
  holding the target for the whole 20 ms control interval; the policy only updates targets at 50 Hz.
- **The loop closes through rewards.** Velocity-tracking reward and penalties are accumulated
  per control step, then converted to advantages via GAE and used by the PPO update in `PPO.py`.

## Project Structure

```
PPO.py                      PPO algorithm (adapted from the repository above, continuous version)
robot/quadruped.xml         Quadruped model (MuJoCo XML, 4 legs × 3 joints = 12 DOF)
config/                     All hyperparameters, annotated (see below)
env/quadruped_env.py        Environment: observations, rewards, termination, domain randomization
control/pd_controller.py    PD joint controller: action → target angles → motor torques
scripts/train_ppo.py        Training loop (calls PPO.py)
scripts/play_ppo.py         Playback + video recording (loads models trained by PPO.py)
scripts/plot_results.py     Training curves (can compare against logs/ref_baseline)
tests/                      Smoke test / zero-action standing test
```

## Quick Start

```bash
# 1. Environment (first time)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

# 2. Verify the environment and model
.venv/bin/python tests/smoke_test.py        # random actions don't crash + PD response
.venv/bin/python tests/zero_action_test.py  # stands for 5 s with zero action

# 3. Train the quadruped with PPO.py
.venv/bin/python scripts/train_ppo.py --iterations 1500 --run-name my_run

# 4. Playback + video + curves
.venv/bin/python scripts/play_ppo.py --checkpoint runs/my_run/best.pt --viewer   # live window
.venv/bin/python scripts/play_ppo.py --checkpoint runs/my_run/best.pt --keyboard # keyboard control
.venv/bin/python scripts/play_ppo.py --checkpoint runs/my_run/best.pt            # record mp4
.venv/bin/python scripts/plot_results.py --logs logs/my_run logs/ref_baseline --labels mine baseline
```

## Configuration

All hyperparameters live in annotated YAML files under `config/`:

| File | Contents |
|---|---|
| `config/robot.yaml` | default pose, joint limits, nominal base height, fall height |
| `config/env.yaml` | simulation timing, command ranges, observation scales/noise, reward weights, termination thresholds, domain randomization |
| `config/pd.yaml` | PD gains (kp, kd), torque limit, action scale |
| `config/train.yaml` | PPO hyperparameters (network, lr, epochs, ε, γ, λ) and training-loop settings (iterations, rollout length, seed, save interval, device) |

**Reward / penalty values** are defined in `config/env.yaml` (the `rewards.weights` block plus the
related thresholds under `rewards:`). The per-term computation and their trigger conditions are
implemented in `env/quadruped_env.py` (`_compute_reward()`), and each term's live per-step value
is exposed via `info["reward_components"]` for debugging.

Each entry in the YAML files is annotated with its meaning and unit.
`scripts/train_ppo.py` reads `config/train.yaml` for defaults; command-line arguments
override them for quick experiments (e.g. `--iterations 300`).

训练默认启用课程学习：初期使用较小速度指令且关闭域随机化，随后在训练前 60% 的进度中
逐步增加到完整难度。训练完成后使用 `--keyboard` 启动遥控：W/S 前后、A/D 横移、
Q/E 转向、空格停止、R 重置。速度步长和上限均在 `config/play.yaml` 中配置。

## Staged training

阶段参数集中在 `config/stages.yaml`。开始新阶段使用 `--init-checkpoint`，它继承上一阶段
Actor、Critic 和优化器动量，但将新阶段 iteration、课程进度和 best score 重新计数；同一阶段
意外中断才使用 `--resume`。

```bash
# 第二阶段：从第一阶段站立模型学习基础前进
.venv/bin/python scripts/train_ppo.py --stage stage2_forward \
  --init-checkpoint runs/locomotion_v2/best.pt --run-name stage2_forward

# 第二阶段中断续训（会从 checkpoint 自动恢复 stage2 配置）
.venv/bin/python scripts/train_ppo.py --resume runs/stage2_forward/last.pt

# 第二阶段达标后进入第三阶段
.venv/bin/python scripts/train_ppo.py --stage stage3_backward_yaw \
  --init-checkpoint runs/stage2_forward/best.pt --run-name stage3_backward_yaw
```

每一阶段持续混入旧任务，并同时设置存活率与成功率门槛。未达到门槛时只保存 `last.pt`，
不会生成误导性的 `best.pt`，此时不应进入下一阶段。
