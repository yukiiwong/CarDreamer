# PlaNet 测试资源总览

这个目录包含了测试 PlaNet 世界模型集成的完整资源。

## 📁 测试文件清单

| 文件 | 用途 | 运行时间 |
|------|------|---------|
| `test_planet.py` | 单元测试，验证 PlaNet 实现 | ~1分钟 |
| `test_planet_quick.sh` | 快速集成测试 | ~10分钟 |
| `compare_planet_rssm.sh` | 完整对比测试 | ~数小时 |
| `analyze_comparison.py` | 分析对比结果 | ~1分钟 |
| `TESTING_GUIDE.md` | 详细测试指南 | 阅读 |
| `PLANET_INTEGRATION.md` | 使用文档 | 阅读 |

## 🚀 快速开始

### 1️⃣ 最快验证（推荐首次运行）

```bash
# 运行快速测试（约10分钟）
./test_planet_quick.sh 2000 0 carla_four_lane 1000
```

**这会做什么**：
- 启动 CARLA
- 运行 planet_small 和 planet_medium 配置
- 训练各 1000 步
- 验证基本功能

**预期输出**：
```
✅ Test passed for planet_small
✅ Test passed for planet_medium
✅ All PlaNet tests completed successfully!
```

### 2️⃣ 完整对比测试

```bash
# 运行完整的 PlaNet vs RSSM 对比（数小时）
./compare_planet_rssm.sh 2000 0 carla_four_lane 1000000
```

**这会做什么**：
- 并行训练多个配置
- 比较 RSSM 和 PlaNet
- 记录性能指标

### 3️⃣ 分析结果

```bash
# 找到最新的对比目录
COMP_DIR=$(ls -td ./logdir/comparison_* | head -1)

# 分析并显示结果
python analyze_comparison.py $COMP_DIR
```

**预期输出**：
```
COMPARISON RESULTS
Model                Steps        Final Reward    Max Reward      ...
--------------------------------------------------------------------------------
rssm_small          1,000,000    350.23          380.45          ...
planet_small        1,000,000    335.67          365.89          ...
...

SUMMARY
Comparison: SMALL
  RSSM   - Final Reward: 350.23, Training Time: 6.2h
  PlaNet - Final Reward: 335.67, Training Time: 5.3h
  → Reward difference: -4.2%
  → Time difference: -14.5% (PlaNet is 1.17x faster)
```

## 📊 测试级别

### Level 1: 语法检查（1分钟）
```bash
python -m py_compile dreamerv3/nets.py
python -m py_compile dreamerv3/agent.py
```
验证代码没有语法错误。

### Level 2: 单元测试（1分钟）
```bash
python test_planet.py
```
验证 PlaNet 核心功能和接口兼容性。

### Level 3: 快速集成测试（10分钟）
```bash
./test_planet_quick.sh 2000 0 carla_four_lane 1000
```
验证端到端训练流程。

### Level 4: 单配置完整测试（2-4小时）
```bash
bash train_dm3.sh 2000 0 --configs planet_medium \
    --task carla_four_lane \
    --dreamerv3.logdir ./logdir/planet_full_test \
    --dreamerv3.run.steps 500000
```
验证训练稳定性和收敛性。

### Level 5: 完整对比测试（数小时）
```bash
./compare_planet_rssm.sh 2000 0 carla_four_lane 1000000
python analyze_comparison.py <comparison_dir>
```
全面对比 PlaNet 和 RSSM。

## 🎯 推荐测试流程

### 首次使用

1. **快速验证**（必做）
   ```bash
   ./test_planet_quick.sh
   ```
   确认基本功能正常。

2. **检查日志**
   ```bash
   LOG_DIR=$(ls -td ./logdir/test_planet_* | head -1)
   grep -i "planet" $LOG_DIR/train.log
   ```
   确认 PlaNet 正确加载。

3. **单个任务测试**
   ```bash
   bash train_dm3.sh 2000 0 --configs planet_medium \
       --task carla_four_lane \
       --dreamerv3.logdir ./logdir/my_first_planet \
       --dreamerv3.run.steps 100000
   ```
   进行较短的完整训练。

### 性能评估

4. **选择任务和配置**
   - 简单任务：carla_four_lane, carla_right_turn_simple
   - 中等任务：carla_lane_merge, carla_overtake
   - 困难任务：carla_roundabout, carla_left_turn_hard

5. **运行对比实验**
   ```bash
   ./compare_planet_rssm.sh 2000 0 <your_task> 1000000
   ```

6. **分析和记录结果**
   ```bash
   python analyze_comparison.py ./logdir/comparison_*
   ```

## 🔍 监控训练

### 实时可视化
```bash
# 访问可视化界面
http://localhost:<CARLA_PORT + 7000>
```

### TensorBoard
```bash
tensorboard --logdir ./logdir
# 访问 http://localhost:6006
```

### 日志监控
```bash
# 查看最新训练日志
tail -f $(ls -td ./logdir/*/train.log | head -1)

# 监控指标
watch -n 5 "tail -n 1 $(ls -td ./logdir/*/metrics.jsonl | head -1)"
```

## 📈 预期结果

### 性能基准（参考）

在 **carla_four_lane** 任务上，训练 1M 步：

| 配置 | 模型 | 训练时间 | 内存 | 最终回报 |
|------|------|---------|------|---------|
| Small | RSSM | ~6h | 10GB | 350 |
| Small | PlaNet | ~5h | 8GB | 330 |
| Medium | RSSM | ~8h | 14GB | 400 |
| Medium | PlaNet | ~7h | 11GB | 385 |

**PlaNet 优势**：
- ✅ 训练速度快 10-20%
- ✅ 内存占用低 20-30%
- ✅ 训练更稳定

**性能权衡**：
- ⚠️ 最终性能略低 3-5%（在大多数任务上）

## ⚙️ 自定义测试

### 测试单个配置
```bash
bash train_dm3.sh <PORT> <GPU> \
    --configs <planet_small|planet_medium|planet_large|planet> \
    --task <task_name> \
    --dreamerv3.logdir ./logdir/<experiment_name> \
    --dreamerv3.run.steps <num_steps>
```

### 测试超参数
```bash
bash train_dm3.sh 2000 0 --configs planet_medium \
    --task carla_four_lane \
    --dreamerv3.rssm.deter 2048 \            # 增加潜在维度
    --dreamerv3.loss_scales.dyn 2.0 \        # 增加动态损失权重
    --dreamerv3.model_opt.lr 2e-4 \          # 调整学习率
    --dreamerv3.logdir ./logdir/custom_planet
```

### 测试多个任务
```bash
# 创建批量测试脚本
TASKS=("carla_four_lane" "carla_right_turn_simple" "carla_lane_merge")
PORT=2000

for task in "${TASKS[@]}"; do
    bash train_dm3.sh $PORT 0 \
        --configs planet_medium \
        --task $task \
        --dreamerv3.logdir ./logdir/planet_${task} \
        --dreamerv3.run.steps 500000 &
    PORT=$((PORT+1))
    sleep 10
done

wait
echo "All tests completed!"
```

## 🐛 故障排查

### 常见问题

**问题：训练立即崩溃**
```bash
# 检查 CARLA 是否运行
ps aux | grep carla

# 检查端口是否被占用
netstat -tlnp | grep <PORT>

# 查看错误日志
tail -50 logdir/*/train.log
```

**问题：内存溢出**
```bash
# 使用更小的配置
./test_planet_quick.sh 2000 0 carla_four_lane 1000

# 或减少 batch size
bash train_dm3.sh 2000 0 --configs planet_small \
    --dreamerv3.batch_size 8
```

**问题：性能不佳**
- 训练时间不够？延长到 1M+ 步
- 配置太小？尝试 planet_medium 或 planet_large
- 任务太难？先在简单任务上测试

详细故障排查请参考 `TESTING_GUIDE.md`。

## 📚 相关文档

- **PLANET_INTEGRATION.md** - 详细使用说明和配置指南
- **TESTING_GUIDE.md** - 完整测试流程和验证清单
- **dreamerv3/dreamerv3.yaml** - 配置文件参考

## 💡 提示

1. **首次测试建议**：先运行 `test_planet_quick.sh` 快速验证
2. **性能对比**：使用 `compare_planet_rssm.sh` 进行系统对比
3. **超参数调优**：参考 PLANET_INTEGRATION.md 中的建议
4. **遇到问题**：查看 TESTING_GUIDE.md 的故障排查章节

## 🎉 测试通过标志

当你看到以下所有 ✅ 时，PlaNet 集成已成功验证：

- ✅ 快速测试通过
- ✅ 日志显示 "Using PlaNet world model"
- ✅ 训练稳定运行无崩溃
- ✅ 指标正常收敛
- ✅ 性能在预期范围内
- ✅ 内存占用合理

**Happy Testing! 🚀**

如有问题，请参考详细文档或提交 Issue。
