# PlaNet 测试指南

本指南提供了测试 PlaNet 集成的完整步骤和检查清单。

## 📋 预备检查

在开始测试前，确保：

- [ ] CARLA 0.9.15 已安装并配置正确
- [ ] 已设置环境变量 `CARLA_ROOT` 和 `PYTHONPATH`
- [ ] 已安装 CarDreamer 和 DreamerV3 依赖
- [ ] GPU 有足够的内存（建议至少 12GB）
- [ ] 有足够的磁盘空间用于日志（每个实验约 5-10GB）

## 🚀 快速验证测试

### 第1步：语法验证

检查代码是否有语法错误：

```bash
# 验证 Python 语法
python -m py_compile dreamerv3/nets.py
python -m py_compile dreamerv3/agent.py

# 如果 JAX 已安装，运行单元测试
python test_planet.py
```

**预期结果**：
- ✅ 没有语法错误
- ✅ 所有测试通过（如果运行了 test_planet.py）

### 第2步：快速训练测试（推荐首先运行）

运行短时间训练确保基本功能正常：

```bash
# 给脚本添加执行权限
chmod +x test_planet_quick.sh

# 运行快速测试（约5-10分钟）
bash test_planet_quick.sh 2000 0 carla_four_lane 1000
```

**参数说明**：
- `2000` - CARLA 端口
- `0` - GPU ID
- `carla_four_lane` - 任务名称
- `1000` - 训练步数（快速测试用）

**检查要点**：
- [ ] CARLA 成功启动
- [ ] 看到 "Using PlaNet world model" 消息
- [ ] 训练开始且没有崩溃
- [ ] 创建了日志目录
- [ ] metrics.jsonl 文件被创建并包含数据

### 第3步：检查日志

查看训练日志确保 PlaNet 正确加载：

```bash
# 查看最新的日志目录
LOG_DIR=$(ls -td ./logdir/test_planet_* | head -1)

# 检查是否有输出
cat $LOG_DIR/train.log | grep -i "planet"

# 查看指标文件
tail -n 5 $LOG_DIR/metrics.jsonl
```

**应该看到**：
```
Using PlaNet world model
Encoder CNN shapes: ...
Encoder MLP shapes: ...
```

## 🔬 完整功能测试

### 测试1：单个配置完整训练

运行一个配置的完整训练（约2-4小时）：

```bash
# 使用 PlaNet medium 配置
bash train_dm3.sh 2000 0 \
    --configs planet_medium \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_test_full \
    --dreamerv3.run.steps 500000
```

**监控**：
- 可视化工具：http://localhost:9000
- TensorBoard：`tensorboard --logdir ./logdir/planet_test_full`

**关键指标检查**：
- [ ] `model_loss` 逐渐下降
- [ ] `eval_return` 逐渐提高
- [ ] 没有 NaN 或 Inf 值
- [ ] GPU 内存稳定（不持续增长）

### 测试2：不同模型大小

测试不同大小的 PlaNet 配置：

```bash
# Small（最快，适合快速迭代）
bash train_dm3.sh 2000 0 --configs planet_small \
    --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/planet_small_test

# Medium（平衡性能和速度）
bash train_dm3.sh 2001 0 --configs planet_medium \
    --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/planet_medium_test

# Large（更好的性能）
bash train_dm3.sh 2002 0 --configs planet_large \
    --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/planet_large_test
```

**比较要点**：
- [ ] Large 模型内存占用更高
- [ ] Small 模型训练速度更快
- [ ] 性能随模型大小提升

### 测试3：不同任务

在不同的驾驶任务上测试 PlaNet：

```bash
# 简单任务
bash train_dm3.sh 2000 0 --configs planet_medium \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_four_lane

# 中等难度
bash train_dm3.sh 2000 0 --configs planet_medium \
    --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/planet_right_turn

# 困难任务
bash train_dm3.sh 2000 0 --configs planet_large \
    --task carla_roundabout \
    --dreamerv3.logdir ./logdir/planet_roundabout
```

## 📊 性能对比测试

### 系统对比：PlaNet vs RSSM

运行完整的对比实验：

```bash
# 给脚本添加执行权限
chmod +x compare_planet_rssm.sh

# 启动对比测试
bash compare_planet_rssm.sh 2000 0 carla_four_lane 1000000
```

这将启动多个训练任务：
- `rssm_small` vs `planet_small`
- `rssm_medium` vs `planet_medium`

**等待训练完成后，分析结果**：

```bash
# 查找最新的对比目录
COMP_DIR=$(ls -td ./logdir/comparison_* | head -1)

# 运行分析脚本
python analyze_comparison.py $COMP_DIR
```

**预期对比结果**：
- [ ] PlaNet 训练速度快 10-15%
- [ ] PlaNet 内存占用低 20-30%
- [ ] PlaNet 最终性能接近 RSSM（90-100%）
- [ ] PlaNet 训练曲线更稳定

## 🔍 深度验证

### 验证1：检查点加载

测试从检查点恢复训练：

```bash
# 先训练一段时间
bash train_dm3.sh 2000 0 --configs planet_medium \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_checkpoint_test \
    --dreamerv3.run.steps 50000

# 等待保存检查点后，停止训练（Ctrl+C）

# 从检查点恢复
bash train_dm3.sh 2000 0 --configs planet_medium \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_checkpoint_test \
    --dreamerv3.run.from_checkpoint ./logdir/planet_checkpoint_test/checkpoint.ckpt
```

**检查要点**：
- [ ] 恢复后继续训练
- [ ] 步数从保存点继续
- [ ] 性能指标连续

### 验证2：超参数调优

测试修改 PlaNet 特定参数：

```bash
# 增加潜在维度
bash train_dm3.sh 2000 0 --configs planet_medium \
    --dreamerv3.rssm.deter 2048 \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_deter2048

# 调整动态损失权重
bash train_dm3.sh 2000 0 --configs planet_medium \
    --dreamerv3.loss_scales.dyn 2.0 \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_dyn2.0
```

### 验证3：可视化检查

在可视化界面中检查 PlaNet 的预测：

1. 访问 http://localhost:9000
2. 检查以下内容：
   - [ ] 观察重建质量良好
   - [ ] 奖励预测合理
   - [ ] 想象的轨迹连贯
   - [ ] 没有明显的模式崩溃

## 📈 性能基准

### 预期性能指标（carla_four_lane，1M steps）

| 配置 | 训练时间 | 内存占用 | 最终回报 | 成功率 |
|------|---------|---------|---------|-------|
| RSSM small | ~6h | 10GB | 350 | 75% |
| PlaNet small | ~5h | 8GB | 330 | 72% |
| RSSM medium | ~8h | 14GB | 400 | 82% |
| PlaNet medium | ~7h | 11GB | 385 | 80% |
| RSSM large | ~12h | 18GB | 430 | 88% |
| PlaNet large | ~10h | 14GB | 415 | 85% |

**注意**：实际数值会因硬件和任务而异。

## 🐛 故障排查

### 问题1：训练立即崩溃

```bash
# 检查错误日志
tail -n 50 logdir/*/train.log

# 常见原因：
# - CARLA 未启动
# - 端口被占用
# - GPU 内存不足
# - 配置文件语法错误
```

**解决方法**：
- 确保 CARLA 正在运行
- 更换端口或清理进程
- 降低模型大小（使用 planet_small）
- 检查 YAML 语法

### 问题2：模型损失为 NaN

```bash
# 查看训练日志
grep -i "nan\|inf" logdir/*/train.log
```

**解决方法**：
- 降低学习率：`--dreamerv3.model_opt.lr 5e-5`
- 增加梯度裁剪：`--dreamerv3.model_opt.clip 500`
- 检查数据是否正常

### 问题3：性能不如预期

**可能原因**：
- 训练步数不够（需要至少 500K-1M 步）
- 超参数不适合任务
- 任务本身较难

**解决方法**：
- 延长训练时间
- 参考文档调整超参数
- 尝试更大的模型

### 问题4：内存溢出

```bash
# 检查 GPU 内存
nvidia-smi
```

**解决方法**：
- 使用更小的配置：`planet_small`
- 减少 batch size：`--dreamerv3.batch_size 8`
- 减少序列长度：`--dreamerv3.batch_length 32`

## ✅ 测试检查清单

完成以下所有测试后，PlaNet 集成即视为验证完成：

### 基础功能
- [ ] 代码语法验证通过
- [ ] 快速训练测试成功
- [ ] 日志正确显示 "Using PlaNet world model"
- [ ] 指标文件正常生成

### 配置测试
- [ ] planet_small 配置工作正常
- [ ] planet_medium 配置工作正常
- [ ] planet_large 配置工作正常
- [ ] 默认 planet 配置工作正常

### 任务测试
- [ ] carla_four_lane 任务测试通过
- [ ] carla_right_turn_simple 任务测试通过
- [ ] 至少一个复杂任务测试通过

### 对比测试
- [ ] PlaNet vs RSSM 对比完成
- [ ] 性能在预期范围内
- [ ] 训练速度有提升
- [ ] 内存占用有降低

### 高级功能
- [ ] 检查点保存/加载正常
- [ ] 可视化界面显示正常
- [ ] 超参数调整生效
- [ ] 多任务稳定性验证

## 📝 报告模板

测试完成后，可以使用以下模板记录结果：

```markdown
# PlaNet 测试报告

## 测试环境
- GPU: [型号]
- CUDA: [版本]
- CarDreamer: [commit hash]
- 测试日期: [日期]

## 测试结果

### 功能测试
- 基础功能：✅/❌
- 配置测试：✅/❌
- 任务测试：✅/❌

### 性能对比
| 配置 | RSSM 回报 | PlaNet 回报 | 性能比 |
|------|----------|------------|--------|
| Small | X | Y | Y/X% |
| Medium | X | Y | Y/X% |

### 训练速度
- RSSM: X steps/sec
- PlaNet: Y steps/sec
- 提升: (Y-X)/X %

### 内存占用
- RSSM: X GB
- PlaNet: Y GB
- 节省: (X-Y)/X %

## 问题记录
[记录遇到的问题和解决方法]

## 结论
[总体评价 PlaNet 的表现]
```

## 🎯 下一步

测试通过后：
1. 根据结果优化超参数
2. 在实际研究任务上使用 PlaNet
3. 考虑向上游项目贡献（可选）
4. 记录最佳实践和经验

## 📚 参考资源

- PlaNet 论文: https://arxiv.org/abs/1811.04551
- CarDreamer 文档: https://car-dreamer.readthedocs.io/
- 集成指南: PLANET_INTEGRATION.md
- 测试脚本: test_planet.py
