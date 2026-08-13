# MuJoCo 四足机械狗 + 从零手写 PPO

用 MuJoCo 自己建模的 12 自由度四足机械狗，用**从零手写的 PPO** 训练行走 + 速度跟踪
（前进 vx / 侧移 vy / 偏航角速度指令）。不依赖 stable-baselines3 等现成 RL 库。

## 算法来源

PPO 算法实现来自强化学习算法教程仓库
**[YWQAQWY/ReinforcementLearningAlgorithm](https://github.com/YWQAQWY/ReinforcementLearningAlgorithm.git)**：

- `PPO.py`（本项目根目录）改编自该仓库的 **PPO 离散动作版**（CartPole-v1，softmax + Categorical）
- 四足狗是连续控制任务，需将策略改为**高斯分布**（Actor 输出均值 μ + log σ，用 `torch.distributions.Normal` 采样）
- 该仓库的 `PPO_continuous.py`（Pendulum-v1 连续版 PPO）可作为改造对照

核心算法为 PPO-Clip + GAE（Schulman et al. 2015/2017）：

```
ratio = exp(logπ_new − logπ_old)
L_CLIP = mean(min(ratio·A, clip(ratio, 1−ε, 1+ε)·A))
loss = −L_CLIP + value_coef·0.5·mean((V−R)²) − entropy_coef·mean(entropy)
```

## 项目结构

```
PPO.py                      PPO 算法（源自上述仓库，改造为连续动作版）
robot/quadruped.xml         四足狗模型（MuJoCo XML，4 腿 × 3 关节 = 12 DOF）
env/quadruped_env.py        环境（完整实现）：PD 控制、观测、奖励、终止、域随机化
ppo/network.py              Actor-Critic 网络（参考实现，完整）
ppo/buffer.py               RolloutBuffer —— 【TODO 1: compute_gae() 等你实现】
ppo/agent.py                PPO 更新 + 观测归一化 —— 【TODO 2: update() 等你实现】
ppo/reference_solution.py   参考答案（建议自己写完再看）
scripts/train_ppo.py        训练主循环（调用 PPO.py，你的算法）
scripts/play_ppo.py         回放 + 录视频（调用 PPO.py 训练出的模型）
scripts/train.py            训练主循环（调用 ppo/ 参考实现，--reference 跑通管线用）
scripts/play.py             回放 + 录视频（参考实现）
scripts/plot_results.py     训练曲线对比（可对比你的实现 vs 参考实现）
tests/                      冒烟测试 / 零动作站立测试
```

## 快速开始

```bash
# 1. 环境（首次）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

# 2. 验证环境和模型
.venv/bin/python tests/smoke_test.py        # 随机动作不崩 + PD 响应
.venv/bin/python tests/zero_action_test.py  # 零动作站立 5 秒

# 3. 用参考实现验证管线（300 迭代约 4 分钟，奖励应上升）
.venv/bin/python scripts/train.py --reference --iterations 300 --run-name ref_check

# 4. 用你自己的 PPO.py 训练四足狗（先把 PPO.py 改成连续动作版）
.venv/bin/python scripts/train_ppo.py --iterations 1500 --run-name my_run

# 5. 回放 + 视频 + 曲线对比
.venv/bin/python scripts/play_ppo.py --checkpoint runs/my_run/best.pt
.venv/bin/python scripts/plot_results.py --logs logs/my_run logs/ref_check --labels 我的 参考
```

## 关键超参数

| 参数 | 值 | 说明 |
|---|---|---|
| 控制频率 | 50 Hz | 模拟 200 Hz，decimation 4 |
| γ / λ | 0.99 / 0.95 | 折扣与 GAE |
| clip ε | 0.2 | PPO 裁剪范围 |
| lr | 1e-3（1500 迭代线性衰减到 0.1 倍） | Adam, eps=1e-5 |
| epochs × batch | 5 × 256 | rollout 2048 步/迭代 |
| entropy_coef | 0.01 | 鼓励探索 |
| value_coef | 1.0 | 值函数损失权重 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| PD (kp, kd) | 20, 0.5 | 力矩限幅 ±33.5 Nm，动作缩放 0.25 rad |
# MuJoCo_Quadruped_Robot
