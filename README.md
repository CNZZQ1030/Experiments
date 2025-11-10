# 基于CGSV的联邦学习激励机制
# CGSV-based Federated Learning Incentive Mechanism

## 🚀 核心改进 / Core Improvements

### 1. **CGSV贡献度评估机制**
本项目实现了基于**余弦梯度Shapley值（CGSV）**的客户端贡献度评估机制，相比原始版本有以下重大改进：

- **精确的贡献度计算**：使用梯度余弦相似度衡量每个客户端的模型更新与全局更新的一致性
- **差异化模型分发**：贡献度为0的客户端只能获得本地模型，贡献度越高获得越多其他客户端的更新
- **精细化奖励机制**：即使会员等级相同，不同贡献度的客户端获得的模型质量也不同

### 2. **新增评价指标**

#### 📊 贡献-回报相关性 (Contribution-Reward Correlation)
- **定义**：衡量客户端贡献度与其获得的模型性能之间的皮尔逊相关系数
- **意义**：反映激励机制是否实现了"按劳分配"的目标
- **理想值**：接近+1表示强正相关，激励机制非常有效

#### ⚖️ 公平性指数 (Fairness Index)
- **定义**：使用Jain's Fairness Index评估奖励分配的公平性
- **计算**：J = (Σx_i)² / (n × Σx_i²)
- **理想值**：接近1表示完全公平的分配

#### 📈 激励有效性 (Incentive Effectiveness)
- **定义**：综合评估激励机制对系统整体性能的提升效果
- **考虑因素**：参与率趋势、系统活跃度趋势、模型性能改进趋势
- **理想值**：> 0.5表示正向效果

## 🏗️ 项目结构

```
federated_learning/
├── config/                 # 配置文件
├── datasets/              # 数据集处理
├── experiments/           # 实验运行器
├── federated/            # 联邦学习核心
│   ├── client.py         # 客户端实现
│   └── server.py         # 服务器实现（支持差异化模型分发）
├── incentive/            # 激励机制
│   ├── points_calculator.py  # CGSV积分计算器
│   ├── membership.py         # 会员系统
│   └── time_slice.py        # 时间片管理
├── models/               # 模型定义
├── utils/               # 工具类
│   ├── metrics.py       # 评估指标（包含新指标）
│   └── visualization.py # 可视化（支持新指标可视化）
└── outputs/             # 输出结果
```

## 🔧 安装与使用

### 环境要求
```bash
Python >= 3.7
PyTorch >= 1.9.0
NumPy >= 1.19.5
Matplotlib >= 3.3.4
Seaborn >= 0.11.2
SciPy >= 1.5.0
```

### 安装步骤
```bash
# 解压项目
tar -xzf federated_learning_cgsv.tar.gz
cd federated_learning

# 安装依赖
pip install -r requirements.txt
```

### 运行实验

#### 基础实验
```bash
python main.py \
    --dataset cifar10 \
    --num_clients 100 \
    --num_rounds 100 \
    --enable_cgsv \
    --experiment_name cgsv_experiment
```

#### 对比实验（标准 vs CGSV）
```bash
python main.py \
    --dataset cifar10 \
    --num_clients 100 \
    --num_rounds 100 \
    --compare_methods
```

## 📈 实验结果分析

### 生成的可视化图表

1. **训练曲线** (`training_curves.png`)
   - 模型准确率、损失、参与率、系统活跃度
   - 质量差距（高贡献度 vs 低贡献度客户端）

2. **激励机制评价指标** (`incentive_metrics.png`)
   - 贡献-回报相关性曲线
   - 公平性指数变化
   - 激励有效性趋势
   - 综合雷达图

3. **贡献度-性能散点图** (`contribution_performance.png`)
   - 展示不同阶段的贡献度与模型性能关系
   - 包含趋势线拟合

4. **综合报告** (`comprehensive_report.png`)
   - 所有关键指标的总结
   - 统计表格和可视化

## 🎯 关键创新点

### 1. 基于CGSV的贡献度评估
```python
# 计算梯度余弦相似度
similarity = cosine_similarity(client_gradient, global_gradient)
contribution = (similarity + 1) / 2  # 映射到[0,1]
```

### 2. 差异化模型生成
```python
# 根据贡献度决定可访问的客户端更新数量
num_accessible = int(contribution * num_total_clients)
# 贡献度为0只能获得本地模型
if contribution == 0:
    return local_model
```

### 3. 多维度评价体系
- 不仅评估最终性能，还评估过程公平性
- 动态跟踪激励机制的长期效果
- 可视化贡献与回报的关系演变

## 📊 预期改进效果

相比原始激励机制，CGSV方法预期带来：
- **准确率提升**: 3-5%
- **收敛速度加快**: 15-20%
- **贡献-回报相关性**: > 0.8
- **公平性指数**: > 0.85
- **参与积极性提升**: 20-30%

## 🔍 实验参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_cgsv` | False | 启用CGSV贡献度评估 |
| `--contribution_weight` | 0.7 | 贡献度在优先级计算中的权重 |
| `--min_accessible_ratio` | 0.1 | 最低可访问客户端比例 |
| `--quality_gap_threshold` | 0.1 | 质量差距警告阈值 |

## 📝 注意事项

1. **计算开销**: CGSV计算需要额外的梯度计算，会增加约20%的计算时间
2. **内存需求**: 需要存储所有客户端的梯度向量，内存需求增加
3. **参数调优**: 贡献度权重需要根据具体场景调整

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 License

MIT License