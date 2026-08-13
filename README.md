# MuJoCo Quadruped Robot

A 12-DOF quadruped robot modeled from scratch in MuJoCo, trained with a **from-scratch PPO
implementation** to walk and track velocity commands (forward vx / lateral vy / yaw rate).
No off-the-shelf RL libraries (stable-baselines3, etc.) are used.

## Algorithm Source

The PPO implementation is adapted from the RL algorithm tutorial repository
**[YWQAQWY/ReinforcementLearningAlgorithm](https://github.com/YWQAQWY/ReinforcementLearningAlgorithm.git)**:

- `PPO.py` (project root) is adapted from the repository's **discrete-action PPO** (CartPole-v1, softmax + Categorical)
- The quadruped is a continuous-control task, so the policy must be changed to a **Gaussian distribution**
  (actor outputs mean μ + log σ, sampled with `torch.distributions.Normal`)
- The repository's `PPO_continuous.py` (continuous PPO for Pendulum-v1) serves as the reference for this adaptation

Core algorithm: PPO-Clip + GAE (Schulman et al. 2015/2017):

```
ratio = exp(logπ_new − logπ_old)
L_CLIP = mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))
loss = −L_CLIP + value_coef·0.5·mean((V−R)²) − entropy_coef·mean(entropy)
```

## Project Structure

```
PPO.py                      PPO algorithm (adapted from the repository above, continuous version)
robot/quadruped.xml         Quadruped model (MuJoCo XML, 4 legs × 3 joints = 12 DOF)
env/quadruped_env.py        Environment (complete): PD control, observations, rewards,
                            termination, domain randomization
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

# 3. Train the quadruped with your own PPO.py (change it to a continuous version first)
.venv/bin/python scripts/train_ppo.py --iterations 1500 --run-name my_run

# 4. Playback + video + curves
.venv/bin/python scripts/play_ppo.py --checkpoint runs/my_run/best.pt
.venv/bin/python scripts/plot_results.py --logs logs/my_run logs/ref_baseline --labels mine baseline
```

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Control frequency | 50 Hz | simulation 200 Hz, decimation 4 |
| γ / λ | 0.99 / 0.95 | discount and GAE |
| clip ε | 0.2 | PPO clipping range |
| lr | 1e-3 (linear decay to 0.1× over 1500 iterations) | Adam, eps=1e-5 |
| epochs × batch | 5 × 256 | rollout 2048 steps per iteration |
| entropy_coef | 0.01 | exploration |
| value_coef | 1.0 | value loss weight |
| max_grad_norm | 1.0 | gradient clipping |
| PD (kp, kd) | 20, 0.5 | torque limit ±33.5 Nm, action scale 0.25 rad |
