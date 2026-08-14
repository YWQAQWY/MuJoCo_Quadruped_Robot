# PPO 四足机器人优化记录

日期：2026-08-13

## 修改目标

本次修改优先解决原实现中会直接影响 PPO 正确性的缺陷，然后增强训练稳定性、评估可信度、断点恢复能力、环境奖励和 sim-to-real 随机化。所有训练行为仍由 `config/` 下的 YAML 控制。

## PPO 核心修改

- 修复 actor loss 的广播错误。旧实现的 `ratio` 为 `[N]`、advantage 为 `[N,1]`，相乘会生成错误的 `[N,N]` 张量；现在全程使用 `[N]`。
- 用 `compute_gae` 重写 GAE。真正终止不 bootstrap，时间上限截断仍从最终状态 bootstrap，但两者都会切断跨 reset 的 GAE 递推。
- 将无界 Gaussian + 环境裁剪改为 tanh-squashed Gaussian，动作始终位于 `(-1,1)`，log-probability 包含 tanh Jacobian 修正，保证训练概率与实际执行动作一致。
- 增加 advantage 标准化、PPO value clipping、Actor/Critic 梯度裁剪、目标 KL 提前停止。
- 网络由单隐藏层 ReLU 升级为两隐藏层 ELU，加入 orthogonal initialization 和小增益 Actor 输出层。
- 增加 `policy_loss`、`value_loss`、`entropy`、`approx_kl`、`clip_fraction`、`explained_var`、`action_std` 等诊断指标。
- Actor 默认学习率由 `1e-3` 调整为 `3e-4`，增加线性学习率退火。

## 训练与 checkpoint 修改

- 分别保存 `terminated` 和 episode boundary，正确处理 MuJoCo 环境的 `terminated/truncated`。
- episode 奖励和长度累计器移到 rollout 循环外，避免跨 rollout 回合的前半段统计丢失。
- 新增固定指令、无噪声、无域随机化的确定性评估。
- `best.pt` 仅在评估奖励提高时更新；`last.pt` 每轮保存。
- checkpoint 新增优化器、iteration、total steps、最佳评估分数、配置和 Python/NumPy/PyTorch RNG 状态。
- 新增 `--resume`，未指定 `--run-name` 时自动继续写入 checkpoint 所在 run。
- 日志 CSV 增加 PPO 诊断、评估结果和当前学习率。

## 环境与奖励修改

- 增加机械功率、支撑足滑移、非足端触地和静止指令姿态奖励项。
- 合法支撑判定同时识别足端球和同一末端刚体上的小腿胶囊，避免默认站姿被错误处罚。
- 增加每回合质量/惯量、PD 增益和 0–2 控制步动作延迟随机化。
- reset 时恢复 PD 默认目标，并在关闭随机化时恢复标称动力学参数。
- viewer 和录像模式现在均会在 `terminated` 或 `truncated` 时重置。

## 工程修改

- 新增 `tests/test_ppo.py`，覆盖 GAE 回合边界、时间截断 bootstrap、动作范围和完整 PPO update。
- smoke test 增加默认站姿不触发非期望接触惩罚的回归检查。
- 固定 requirements 中已验证依赖版本。
- README 更新动作分布、断点恢复、best/last checkpoint 语义。
- 训练曲线滑动平均支持短 run 和 NaN，不再因窗口大于日志长度而产生维度错误。

## 验证结果

- `tests/test_ppo.py`：通过。
- `tests/smoke_test.py`：通过；随机动作 500 步无 NaN/Inf，PD 响应方向正确。
- `tests/zero_action_test.py`：通过；零动作稳定站立 5 秒，最大 roll 约 0.01°、最大 pitch 约 1.89°、最低基座高度约 0.331 m。
- Python 语法编译检查：通过。
- 2 iteration × 128 steps CPU 短训练：通过，成功生成 `last.pt` 和 `best.pt`。
- 从 `last.pt` 恢复并继续第 3 iteration：通过。
- `git diff --check`：通过。

## 兼容性说明

策略和价值网络从一层改为两层，因此旧的 `runs/my_run/*.pt` 和 `runs/ref_baseline/*.pt` 无法直接加载到新网络。需要用新实现重新训练。旧 CSV 仍可用于曲线对比，但旧日志没有新加入的所有诊断字段。

## 推荐正式训练命令

```bash
.venv/bin/python scripts/train_ppo.py --iterations 1500 --run-name ppo_v2
```

断点续训：

```bash
.venv/bin/python scripts/train_ppo.py --iterations 1500 --resume runs/ppo_v2/last.pt
```

## 2026-08-13：配置集中化与模块化

- `train.yaml` 新增网络结构、激活函数、初始化增益、初始方差、方差边界、数值稳定系数、评估指令和日志周期；PPO 不再在训练入口中依赖这些硬编码值。
- 训练、评估与回放各自的观测噪声、域随机化、外部指令和确定性策略开关也全部进入配置。
- 新增 `play.yaml`，统一管理回放种子、时长、FPS、viewer 周期、关闭等待、渲染尺寸、编码器、相机、渲染后端和速度指令脚本。
- 新增 `plot.yaml`，统一管理平滑窗口、画布尺寸、DPI、网格透明度和 explained variance 坐标范围。
- `env.yaml` 新增静止指令阈值和初始高度偏移。
- `robot.yaml` 新增 XML 路径、躯干/地面名称、腿名称和足端命名规则；环境维度及关节限位重复次数现在由机器人配置推导。
- PPO 网络改为按 `hidden_dims` 动态构建任意层数，并支持 `relu`、`elu`、`tanh`。
- 配置加载器增加必填字段、网络激活和机器人关节数量一致性校验，并提供 `load_all()`。
- checkpoint 现在包含 `config/` 下所有 YAML 的快照，训练命令行覆盖值会写入实际生效的训练配置。
- CLI 仍保留作为临时覆盖层，因此无需修改 YAML 也能做快速实验；默认值全部来自 `config/`。

## 2026-08-14：稳定行走课程学习与键盘控制

- 新增训练课程学习：从 20% 指令范围、0% 域随机化开始，在训练前 60% 进度内平滑提升到完整难度。
- 课程同步调节速度范围、静止指令概率、初始姿态扰动、摩擦、质量/惯量、PD 增益、动作延迟和随机推搡。
- CSV 新增 `command_scale` 和 `randomization_scale`，便于对照训练曲线分析难度变化。
- 回放新增 `--keyboard` 模式：W/S 前后、A/D 横移、Q/E 转向、空格停止、R 重置。
- 键位、速度增量和最大速度全部位于 `config/play.yaml`。
- 新增无 GUI 测试，覆盖课程调度起止值和键盘速度命令。

## 2026-08-14：打破站立局部最优

- 观测加入机体坐标系基座线速度，维度由 45 增加到 48，使策略可以直接形成速度误差闭环。
- 指令课程从纯前进 `vx=[0,0.5]、vy=0、yaw=0` 开始，再连续插值到完整全方向范围。
- 静止指令概率降低到 0.1；非零移动指令下实际速度过低时增加 `no_motion` 惩罚。
- 线速度和角速度 tracking sigma 分离并收紧，tracking 权重提高，高度固定奖励降低。
- 减弱 action rate、torque、dof acceleration、power 和 feet slip 的初期惩罚，允许先形成周期步态。
- entropy 改为 tanh 后有界动作分布的采样估计；entropy coefficient 降至 0.001，初始 log std 为 -1.5，上限为 0。
- Actor/Critic 学习率退火保留 `3e-5/1e-4` 下限，不再在训练末期降为零。
- 评估增加线速度误差、转向误差、存活率、成功率与 tracking score；`best.pt` 需满足 90% 存活率并按 tracking score 选取。
- 新网络输入维度为 48，旧的 45 维 checkpoint 不再兼容，需要重新训练。
