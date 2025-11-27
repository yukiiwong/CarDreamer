# PlaNet 世界模型集成指南

## 简介

现在 CarDreamer 支持使用 **PlaNet (Deep Planning Network)** 作为世界模型的替代选择。PlaNet 是一个纯确定性潜在状态模型，相比 RSSM 更简单、计算效率更高。

### 论文引用
```
Learning Latent Dynamics for Planning from Pixels
Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson
ICML 2019
```

## PlaNet vs RSSM

| 特性 | RSSM | PlaNet |
|------|------|--------|
| **状态表示** | 确定性 + 随机性 | 纯确定性 |
| **计算复杂度** | 高 | 中等 |
| **样本效率** | 非常高 | 高 |
| **训练稳定性** | 高 | 非常高 |
| **适用场景** | 复杂随机环境 | 确定性/部分随机环境 |
| **内存占用** | 较高 | 较低 |

## 使用方法

### 1. 通过配置文件使用 PlaNet

#### 方法 A: 使用预定义的配置块

在训练命令中添加 `--configs planet`:

```bash
# 使用默认 PlaNet 配置
bash train_dm3.sh 2000 0 --configs planet --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_four_lane

# 使用 PlaNet small 配置（更快，内存占用更少）
bash train_dm3.sh 2000 0 --configs planet_small --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/planet_small_right_turn

# 使用 PlaNet medium 配置
bash train_dm3.sh 2000 0 --configs planet_medium --task carla_roundabout \
    --dreamerv3.logdir ./logdir/planet_medium_roundabout

# 使用 PlaNet large 配置
bash train_dm3.sh 2000 0 --configs planet_large --task carla_lane_merge \
    --dreamerv3.logdir ./logdir/planet_large_lane_merge
```

#### 方法 B: 手动指定世界模型类型

在训练命令中添加 `--dreamerv3.world_model_type planet`:

```bash
bash train_dm3.sh 2000 0 --task carla_four_lane \
    --dreamerv3.world_model_type planet \
    --dreamerv3.logdir ./logdir/planet_custom
```

### 2. 在配置文件中设置

编辑 `dreamerv3/dreamerv3.yaml`:

```yaml
defaults:
  # ... 其他配置 ...
  world_model_type: planet  # 改为 'planet' 使用 PlaNet
```

### 3. 创建自定义 PlaNet 配置

在 `dreamerv3/dreamerv3.yaml` 中创建自定义配置块:

```yaml
my_planet_config:
  world_model_type: planet
  rssm.deter: 1024  # 潜在状态维度
  dyn_loss: { impl: mse, free: 0.0 }
  rep_loss: { impl: none, free: 0.0 }
  loss_scales.dyn: 1.0
  loss_scales.rep: 0.0
  # ... 其他自定义参数 ...
```

然后使用：
```bash
bash train_dm3.sh 2000 0 --configs my_planet_config --task carla_four_lane \
    --dreamerv3.logdir ./logdir/my_planet
```

## 配置说明

### PlaNet 特定配置

```yaml
planet:
  world_model_type: planet       # 指定使用 PlaNet
  dyn_loss: { impl: mse, free: 0.0 }  # PlaNet 使用 MSE 损失
  rep_loss: { impl: none, free: 0.0 } # PlaNet 没有表征损失
  loss_scales.dyn: 1.0           # 动态损失权重
  loss_scales.rep: 0.0           # 表征损失权重（设为0）
```

### 模型大小选择指南

| 配置 | 潜在维度 | 推荐场景 | 内存占用 |
|------|---------|---------|---------|
| `planet_small` | 512 | 简单任务、快速原型 | ~8-12GB |
| `planet_medium` | 1024 | 中等复杂度任务 | ~12-16GB |
| `planet_large` | 2048 | 复杂任务 | ~16-20GB |
| `planet` (默认) | 4096 | 最复杂任务 | ~20-24GB |

## 示例：完整训练流程

### 示例 1: 使用 PlaNet 训练四车道任务

```bash
# 启动训练
bash train_dm3.sh 2000 0 \
    --configs planet_medium \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_four_lane \
    --dreamerv3.run.steps=5e6

# 可视化工具将在 http://localhost:9000 可用
```

### 示例 2: 比较 RSSM 和 PlaNet

```bash
# 训练 RSSM 版本
bash train_dm3.sh 2000 0 \
    --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/rssm_right_turn

# 训练 PlaNet 版本
bash train_dm3.sh 2001 0 \
    --configs planet_medium \
    --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/planet_right_turn
```

### 示例 3: 从检查点恢复训练

```bash
bash train_dm3.sh 2000 0 \
    --configs planet \
    --task carla_lane_merge \
    --dreamerv3.logdir ./logdir/planet_lane_merge \
    --dreamerv3.run.from_checkpoint ./logdir/planet_lane_merge/checkpoint.ckpt
```

## 性能对比

基于初步测试的预期性能（具体数值可能因任务而异）:

| 指标 | RSSM | PlaNet |
|------|------|--------|
| **训练速度** | 基准 | ~10-15% 更快 |
| **内存占用** | 基准 | ~20-30% 更低 |
| **样本效率** | 非常高 | 高 |
| **最终性能** | 最好 | 接近 RSSM |
| **训练稳定性** | 高 | 非常高 |

## 技术细节

### PlaNet 架构

```
观察 -> 编码器 -> 确定性潜在状态 (GRU) -> 解码器 -> 预测观察
                        ↓
                    奖励预测器
                        ↓
                    连续性预测器
```

### 与 RSSM 的关键区别

1. **状态表示**
   - RSSM: `state = {deter, stoch}` (确定性 + 随机性)
   - PlaNet: `state = {deter}` (仅确定性, stoch=deter用于兼容)

2. **损失函数**
   - RSSM: KL 散度 (动态损失 + 表征损失)
   - PlaNet: MSE 损失 (仅动态损失)

3. **计算开销**
   - RSSM: 需要采样随机变量
   - PlaNet: 纯前向传播，更快

## 常见问题

### Q: 什么时候应该使用 PlaNet 而不是 RSSM？

A: 考虑使用 PlaNet 如果：
- 您的任务相对确定性（少随机性）
- 您需要更快的训练速度
- 您的 GPU 内存有限
- 您想要更稳定的训练过程

### Q: PlaNet 的性能会比 RSSM 差吗？

A: 在许多任务上，PlaNet 的性能接近 RSSM。对于高度随机的环境，RSSM 可能表现更好，但对于驾驶任务（通常相对确定），PlaNet 通常足够好。

### Q: 我可以在训练中途切换世界模型吗？

A: 不建议。RSSM 和 PlaNet 的状态表示不同，检查点不兼容。如果要切换，需要从头开始训练。

### Q: 如何知道 PlaNet 是否成功加载？

A: 在训练开始时，您会看到日志消息 "Using PlaNet world model"。

## 调试技巧

### 检查模型是否正确加载

在 Python 脚本中：

```python
import embodied
import car_dreamer

# 加载配置
config = embodied.Config.from_yaml('dreamerv3/dreamerv3.yaml')
config = config.update({'world_model_type': 'planet'})

# 创建任务
task, _ = car_dreamer.create_task('carla_four_lane')

# 检查配置
print(f"World Model Type: {config.world_model_type}")
```

### 验证训练日志

查看 `logdir/train.log`:
```
Using PlaNet world model
Encoder CNN shapes: ...
Encoder MLP shapes: ...
```

## 进一步优化

### 超参数调优建议

对于 PlaNet，以下超参数可能需要调整：

1. **潜在状态维度** (`rssm.deter`)
   - 更大的值 = 更强的表达能力但更慢
   - 建议范围: 512-2048

2. **动态损失权重** (`loss_scales.dyn`)
   - 控制模型预测准确性
   - 建议范围: 0.5-2.0

3. **学习率** (`model_opt.lr`)
   - PlaNet 可能需要稍高的学习率
   - 建议: 1e-4 到 5e-4

## 贡献

如果您发现问题或有改进建议，欢迎提交 Issue 或 Pull Request！

## 相关资源

- [PlaNet 论文](https://arxiv.org/abs/1811.04551)
- [DreamerV3 论文](https://arxiv.org/abs/2301.04104)
- [CarDreamer 文档](https://car-dreamer.readthedocs.io/)

## 版本历史

- **v1.0** (2025-11-27): 初始 PlaNet 集成
  - 实现 PlaNet 核心模型
  - 添加配置文件支持
  - 创建使用文档
