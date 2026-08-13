# MuJoCo Quadruped Robot

A 12-DOF quadruped robot modeled from scratch in MuJoCo, trained with a **from-scratch PPO
implementation** to walk and track velocity commands (forward vx / lateral vy / yaw rate).
No off-the-shelf RL libraries (stable-baselines3, etc.) are used.

## Algorithm Source

The PPO implementation is adapted from the RL algorithm tutorial repository
**[YWQAQWY/ReinforcementLearningAlgorithm](https://github.com/YWQAQWY/ReinforcementLearningAlgorithm.git)**:

- `PPO.py` (project root) started from the repository's **discrete-action PPO** (CartPole-v1, softmax + Categorical)
- It has been converted to a **bounded continuous Gaussian policy** for the quadruped: the actor outputs
  the mean μ and a learnable log σ, samples are transformed by `tanh`, and PPO log-probabilities include
  the corresponding Jacobian correction
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
  │  obs (45-d) ──► PolicyNet (Gaussian) ──► sample action a ∈ R¹²        │
  │                [base ang vel, projected  [tanh-bounded to [-1, 1]]     │
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
              └──► next obs (45-d) ──► back to policy
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

# 3. Train the quadruped with your PPO.py
.venv/bin/python scripts/train_ppo.py --iterations 1500 --run-name my_run

# Resume a complete checkpoint (optimizer/RNG/step state are restored)
.venv/bin/python scripts/train_ppo.py --iterations 1500 --resume runs/my_run/last.pt

# 4. Playback + video + curves
.venv/bin/python scripts/play_ppo.py --checkpoint runs/my_run/best.pt --viewer   # live window
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
| `config/play.yaml` | 回放种子、速度指令脚本、视频尺寸/编码器/相机和 viewer 日志周期 |
| `config/plot.yaml` | 曲线平滑窗口、画布尺寸、DPI、网格和坐标范围 |

`config.load()` 会在读取时校验关键字段；`config.load_all()` 用于将全部 YAML 配置快照写入 checkpoint。
训练和回放的常用命令行参数仍可覆盖 YAML，checkpoint 保存的是覆盖后的实际训练配置。

Training writes `last.pt` every iteration and updates `best.pt` only when the fixed-command,
noise-free evaluation score improves. New checkpoints also contain optimizer state, configuration,
step counters, and RNG state. Checkpoints created by the previous one-hidden-layer policy are not
architecture-compatible with the current two-hidden-layer network.

Each entry in the YAML files is annotated with its meaning and unit.
`scripts/train_ppo.py` reads `config/train.yaml` for defaults; command-line arguments
override them for quick experiments (e.g. `--iterations 300`).
