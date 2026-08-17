# PPO 四足机器人修改记录

## 2026-08-17：第二阶段前进训练修复

### 根因

- `stage2_forward_gait/last.pt` 实际前进速度只有约 `0.001–0.002 m/s`，足端接触率接近 100%，策略收敛为稳定站立。
- 原二值步态奖励使四脚着地仍获得一半匹配奖励；`vx=0.2 m/s` 也可能被静止策略按绝对误差阈值误判成功。
- `contact_force_threshold` 原先未被使用，课程难度只随 iteration 增长，新增相位输入不足以让继承的站立策略主动抬脚。

### 实现

- 增加参数化对角小跑参考轨迹和残差控制：`q_target = q_default + q_reference + action_scale × residual_scale × action_ppo`。静止指令下 `q_reference=0`。
- 使用 MuJoCo 法向接触力、进入/释放双阈值和足端/腿部/躯干分类；评估信息新增足端接触率、腾空率、对角接触率和机身接触次数。
- 将步态匹配改为平滑相位约束：摆动腿触地、支撑腿失联、摆动脚高度不足和高速落脚均为代价。四脚全部着地不再获得正步态奖励。
- 第二阶段改为能力驱动课程，连续两次满足门槛才提升指令范围、残差比例和随机化强度；初期关闭随机推搡和动力学随机化。
- 评估增加方向、最低速度比例和静止漂移条件。前进指令要求实际速度至少达到目标的 60%，因此静止策略不能通过 `0.2 m/s`。
- checkpoint 增加课程状态，并继续保存完整配置；回放程序从 checkpoint 恢复环境配置，确保参考步态和残差控制不会在键盘回放时丢失。
- 新增 `scripts/validate_reference_gait.py`。训练入口会核对当前参考参数是否存在于合格扫描报告，未通过时拒绝开始第二阶段训练。

### 参考步态验证结果

- 扫描 384 组参数，28 组通过全部硬门槛。
- 当前默认组：频率 `1.8 Hz`、占空比 `0.52`、髋摆幅 `-0.15 rad`、膝周期摆幅 `-0.1 rad`、摆动屈膝偏置 `+0.45 rad`、相位偏置 `-0.8 rad`。
- 10 秒无摔倒，平均前进速度 `0.615 m/s`，单脚腾空率约 `42%–47%`，四脚同时着地率 `1.2%`，躯干接触为 0。
- 完整报告：`logs/reference_gait_scan.json`。

### 使用命令

```bash
# 1. 修改参考参数后必须重新扫描
.venv/bin/python scripts/validate_reference_gait.py

# 2. 从第一阶段合格站立模型开始第二阶段
.venv/bin/python scripts/train_ppo.py --stage stage2_forward \
  --init-checkpoint runs/locomotion_v2/best.pt --run-name stage2_forward_residual

# 3. 中断后恢复（课程等级也会恢复）
.venv/bin/python scripts/train_ppo.py --resume runs/stage2_forward_residual/last.pt

# 4. 仅在生成 best.pt 后进行键盘回放
.venv/bin/python scripts/play_ppo.py \
  --checkpoint runs/stage2_forward_residual/best.pt --keyboard
```

### 兼容性与验收

- 旧 45/48/54 维 checkpoint 仍可回放；阶段初始化继续支持将 48 维输入层扩展为 54 维，新列置零。
- 第二阶段 `best.pt` 要求存活率至少 95%，所有 `0/0.2/0.3/0.5 m/s` 指令均通过严格评估，并连续达标两次。
- 已通过 `test_curriculum_keyboard.py`、`test_ppo.py`、`smoke_test.py` 和一次 32 步完整训练入口测试。
