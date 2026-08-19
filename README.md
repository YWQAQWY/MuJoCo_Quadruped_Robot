# MuJoCo Quadruped Robot

A 12-DOF quadruped robot modeled from scratch in MuJoCo, trained with a **from-scratch PPO
implementation** to stand, walk and track velocity commands (forward vx / lateral vy / yaw rate).
No off-the-shelf RL libraries (stable-baselines3, etc.) are used.

## Current Status (2026-08-19)

| Stage | Capability | Model |
| --- | --- | --- |
| 1 — Stand | Stable standing | `runs/locomotion_v2/best.pt` |
| 2 — Forward walk | Walk forward at 0.2/0.3 m/s and stop, straight-line drift ≤ 0.03 rad/s | `runs/stage2_straight_walk_v6/best.pt` |
| 3–5 — Backward/turn, lateral, robustness | Not started | — |

Keyboard playback of the stage-2 model (arrow keys ↑/↓ forward/backward, ←/→ turn, Q/E lateral,
Space stop, R reset):

```bash
.venv/bin/python scripts/play_ppo.py \
  --checkpoint runs/stage2_straight_walk_v6/best.pt --keyboard
```

## Algorithm Source

The PPO implementation is adapted from the RL algorithm tutorial repository
**[YWQAQWY/ReinforcementLearningAlgorithm](https://github.com/YWQAQWY/ReinforcementLearningAlgorithm.git)**:

- `PPO.py` (project root) started from the repository's **discrete-action PPO** (CartPole-v1, softmax + Categorical).
- It has been converted to a **tanh-squashed Gaussian policy** for the quadruped: the actor outputs
  mean μ and learnable log σ, and both actions and log-probabilities use the same bounded transform.
  The entropy bonus estimates entropy after tanh instead of rewarding unbounded pre-tanh variance.
- The repository's `PPO_continuous.py` (continuous PPO for Pendulum-v1) was the reference for this conversion.

Core algorithm: PPO-Clip + GAE (Schulman et al. 2015/2017):

```
ratio = exp(logπ_new − logπ_old)
L_CLIP = mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))
loss = −L_CLIP + value_coef·0.5·mean((V−R)²) − entropy_coef·mean(entropy)
```

## Control Chain

How the trained policy drives the robot, end to end:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  PPO policy (runs at 50 Hz in PPO.py)                                  │
  │                                                                        │
  │  obs (57-d) ──► PolicyNet (tanh Gaussian) ──► action a ∈ R¹²          │
  │                [base ang vel, projected  [clipped to [-1, 1]]          │
  │                 gravity, commands, joint                               │
  │                 pos/vel, previous action,                              │
  │                 phase, contacts, heading,                              │
  │                 filtered yaw rate]                                     │
  └───────────┬──────────────────────────────────────────┬─────────────────┘
              │ action a (12 joint-target offsets)       │ obs (next state)
              ▼                                          │
  ┌──────────────────────────────────────────────────────┴─────────────────┐
  │  env/quadruped_env.py + control/pd_controller.py (50 Hz)               │
  │                                                                        │
  │  1. target joint angles:  q_target = q_default + q_reference           │
  │                            + action_scale · residual_scale · a         │
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
              └──► next obs (57-d) ──► policy
```

- **Policy outputs actions, not torques.** Each of the 12 actions is a joint-target offset added
  to the default pose and the reference trot (`q_reference`), scaled by `action_scale × residual_scale`
  (stage 2: 0.4 × the curriculum's residual ratio) — so the policy "thinks" in joint-angle space.
  The residual ratio grows with the curriculum (2% → 5% → 10% → 20%), letting the policy first
  learn to preserve the open-loop trot, then earn fine-grained control.
- **PD turns angles into torques.** The PD law lives in `control/pd_controller.py`
  (`set_action` + `compute_torques`); the env calls it every simulation step (200 Hz),
  holding the target for the whole 20 ms control interval; the policy only updates targets at 50 Hz.
- **The loop closes through rewards.** Velocity-tracking rewards and penalties are accumulated
  per control step, then converted to advantages via GAE and used by the PPO update in `PPO.py`.

### Observation Space (57-d)

| Block | Dims | Content |
| --- | --- | --- |
| Base linear velocity | 3 | Body-frame vx, vy, vz |
| Base angular velocity | 3 | Body-frame ωx, ωy, ωz |
| Projected gravity | 3 | Gravity direction in body frame |
| Commands | 3 | vx, vy, yaw-rate targets |
| Joint positions | 12 | q − q_default |
| Joint velocities | 12 | q̇ |
| Previous action | 12 | Last policy output |
| Gait phase | 2 | sin/cos of the reference trot phase |
| Foot contacts | 4 | FL/FR/RL/RR ground-contact flags |
| Heading error | 2 | sin/cos of accumulated heading vs. episode start |
| Filtered yaw rate | 1 | Low-pass (EMA, τ≈1 s) yaw rate — isolates persistent drift from the 1.8 Hz trot oscillation |

The heading error and filtered yaw rate were added specifically to make the straight-walking
penalties observable and optimizable (see `docs/reward_design.md`). Playback auto-detects the
observation dimension (45/48/54/57) from the checkpoint and builds the matching environment;
older checkpoints remain playable.

## Rewards

Reward/penalty weights live in `config/env.yaml` (`rewards.weights` plus related thresholds).
Key groups:

- **Tracking**: `lin_vel` Gaussian velocity tracking, `ang_vel` yaw tracking, `base_height`
- **Speed error**: symmetric linear `speed_error` — overshooting is penalized as much as
  undershooting, so a uniform speed bias cannot pay off
- **Straightness**: `straight_heading` (squared accumulated heading error) plus `yaw_drift`
  (linear penalty on the low-pass filtered yaw rate — immediate and observable, and it does not
  punish the trot's intrinsic yaw oscillation)
- **Gait quality** (stage 2, moving only): diagonal-trot contact matching, swing-foot clearance,
  air time, touchdown velocity
- **Safety/energy**: joint limits, body contacts, foot slip, power, action rate, termination

Each term's per-step value is exposed via `info["reward_components"]` for debugging.
The full design rationale and the debugging history are documented in
**[docs/reward_design.md](docs/reward_design.md)**.

## Project Structure

```
PPO.py                      PPO algorithm (adapted from the repository above, continuous version)
robot/quadruped.xml         Quadruped model (MuJoCo XML, 4 legs × 3 joints = 12 DOF)
config/                     All hyperparameters, annotated (see below)
env/quadruped_env.py        Environment: observations, rewards, termination, domain randomization
control/pd_controller.py    PD joint controller: action → target angles → motor torques
scripts/train_ppo.py        Training loop (stages, curriculum, evaluation gates)
scripts/play_ppo.py         Playback, video recording, keyboard control
scripts/validate_reference_gait.py  Reference-trot feasibility scan + speed-profile preflight
scripts/plot_results.py     Training curves (can compare against logs/ref_baseline)
docs/reward_design.md       Reward/penalty design rationale and debugging history
tests/                      Unit tests / smoke test / zero-action standing test
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

# 3. Train the quadruped with PPO
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
| --- | --- |
| `config/robot.yaml` | default pose, joint limits, nominal base height, fall height |
| `config/env.yaml` | simulation timing, command ranges, observation scales/noise, reward weights, termination thresholds, domain randomization |
| `config/pd.yaml` | PD gains (kp, kd), torque limit, action scale |
| `config/train.yaml` | PPO hyperparameters (network, lr, epochs, ε, γ, λ) and training-loop settings (iterations, rollout length, seed, save interval, device) |
| `config/stages.yaml` | staged tasks, competence curriculum, residual scale and stage gates |
| `config/play.yaml` | playback, keyboard mapping (keys, step sizes, speed limits), video recording |
| `config/gait_scan.yaml` | reference-trot feasibility scan, physical thresholds and parameter grid |
| `config/expert.yaml` | DogML retargeting, MuJoCo replay filters and behavior-cloning settings |

Each YAML entry is annotated with its meaning and unit. `scripts/train_ppo.py` reads
`config/train.yaml` for defaults; command-line arguments override them for quick experiments
(e.g. `--iterations 300`).

Training uses curriculum learning by default: it starts with small velocity commands and no
domain randomization, then ramps both up over the first 60% of the run.

## Staged Training

Stages are defined in `config/stages.yaml`:

| Stage | Goal | Command range |
| --- | --- | --- |
| 1 `stand` | Stable standing | — |
| 2 `stage2_forward` | Stand + forward walk + stop (reference trot + PPO residual) | vx ∈ [0.2, 0.3] |
| 3 `stage3_backward_yaw` | Add backward and turning | vx ∈ [−0.2, 0.5], yaw ∈ [−0.2, 0.2] |
| 4 `stage4_lateral_combined` | Add lateral and combined commands | full 2-D + yaw |
| 5 `stage5_robust` | Full commands + full randomization + push recovery | full range |

Start a new stage with `--init-checkpoint`: it inherits the previous stage's Actor, Critic and
optimizer momentum but restarts the iteration counter, curriculum progress and best score.
Use `--resume` only to continue the same stage after an interruption.

```bash
# The reference trot must pass the feasibility scan before stage-2 training
.venv/bin/python scripts/validate_reference_gait.py

# Stage 2: learn forward walking from the stage-1 standing model
.venv/bin/python scripts/train_ppo.py --stage stage2_forward \
  --init-checkpoint runs/locomotion_v2/best.pt --run-name stage2_forward

# Resume the same stage after an interruption
.venv/bin/python scripts/train_ppo.py --resume runs/stage2_forward/last.pt

# Stage 3 after stage 2 passes its gates
.venv/bin/python scripts/train_ppo.py --stage stage3_backward_yaw \
  --init-checkpoint runs/stage2_forward/best.pt --run-name stage3_backward_yaw
```

Stage 2 uses `default pose + reference diagonal trot + PPO residual`. The reference gait is
disabled under stand commands. The competence curriculum only advances after **two consecutive
evaluations** meet the survival and per-command gates; each stage keeps a fraction of the
previous stage's tasks in the mix.

### Evaluation Gates and Checkpoints

- Evaluation is deterministic (fixed commands, no noise, no domain randomization).
- `best.pt` requires survival ≥ 95% and 100% strict success on all commands twice in a row
  (speed ratio ≥ 80%, linear error ≤ 0.12 m/s, heading drift ≤ 0.03 rad/s, yaw oscillation
  ≤ 0.5 rad/s). Until then only `last.pt` is saved — do not proceed to the next stage without
  a `best.pt`.
- Checkpoints store the full config snapshot, optimizer/RNG state and the curriculum state,
  so both `--resume` and playback reproduce the exact training setup.
- **Residual scale**: training, evaluation and playback must all use the same residual scale.
  The checkpoint stores `curriculum_residual_scale` and playback restores it (old checkpoints
  derive it from their curriculum level). Without this, the default scale (0.2) amplifies a
  level-0 policy's residual by 10× — this bug once manifested as a 100° crouch at stand and
  wild drift during playback.

## Keyboard Control

Run playback with `--keyboard`, click the MuJoCo viewer window to give it focus, then:

| Key | Command |
| --- | --- |
| ↑ / ↓ | forward / backward (0.2 m/s per press) |
| ← / → | turn left / right |
| Q / E | lateral left / right |
| Space | stop (zero command) |
| R | reset |

Stage 2 only trained forward walking and stopping; backward/turning require the stage-3 model
and lateral movement requires stage 4 — unresponsive directions are untrained capabilities,
not key-binding issues. Key names, step sizes and speed limits are configured in
`config/play.yaml` (`max_vx` is clamped to the trained range).

## DogML Expert-Data Pipeline

DogML's 191-d motion representation cannot be used as PPO actions directly. It is first
retargeted and replayed in the current MuJoCo model; behavior cloning is only allowed once the
strict filters pass and the minimum sample count is reached:

```bash
.venv/bin/python scripts/prepare_expert_dataset.py
.venv/bin/python scripts/pretrain_expert_bc.py
```

If the first step does not produce `dataset/processed/walk_expert.npz`, the data skeleton is
incompatible with the current robot: do not lower the thresholds to force training — improve the
skeleton calibration or use state-action data that matches the current model instead.
