# 基于稀疏化的联邦学习激励机制 / Sparsification-based Federated Learning Incentive Mechanism

## 📖 概述 / Overview

本项目实现了一种新颖的联邦学习激励机制，使用**模型稀疏化（减法）**替代传统的**选择性聚合（加法）**方法。核心思想是根据客户端的贡献度对全局模型进行不同程度的稀疏化处理。

This project implements a novel federated learning incentive mechanism using **model sparsification (subtraction)** instead of traditional **selective aggregation (addition)** methods. The core idea is to apply different levels of sparsification to the global model based on client contributions.

## 🎯 核心创新 / Core Innovation

### 传统方法（加法）vs 新方法（减法）

**传统方法 (UPSM - 加法策略):**
- 选择性地聚合部分客户端的更新
- 使用Boltzmann分布进行概率采样
- 高贡献客户端的更新被优先选择

**新方法 (稀疏化 - 减法策略):**
- 所有客户端更新都参与聚合（使用FedAvg）
- 对聚合后的全局模型进行差异化稀疏处理
- 高贡献客户端获得更完整的模型（低稀疏率）
- 低贡献客户端获得稀疏化的模型（高稀疏率）

### 稀疏化算法设计

#### 1. 保留率计算 / Keep Ratio Calculation

```
α_i = Min_Keep + (1 - Min_Keep) × (r_i)^λ
```

- `r_i`: 客户端i的归一化贡献排名 (0到1)
- `λ`: 调节系数（λ>1为凸函数，让高贡献者优势更明显）
- `Min_Keep`: 最低保留率（如0.1，即保留10%参数）

#### 2. 会员等级稀疏率范围 / Membership Level Sparsity Ranges

| 等级/Level | 贡献度范围/Contribution | 稀疏率范围/Sparsity Range | 保留参数/Keep Params |
|------------|-------------------------|---------------------------|----------------------|
| Diamond    | Top 10% (r_i > 0.9)     | [0%, 10%]                | 90%-100%            |
| Gold       | Next 30%                | [10%, 30%]               | 70%-90%             |
| Silver     | Next 40%                | [30%, 60%]               | 40%-70%             |
| Bronze     | Bottom 20%              | [60%, 95%]               | 5%-40%              |

## 🚀 快速开始 / Quick Start

### 环境要求 / Requirements

```bash
# 安装依赖 / Install dependencies
pip install -r requirements.txt
```

### 基础实验 / Basic Experiment

```bash
# MNIST数据集，IID分布，magnitude稀疏化
python main_sparsification.py --dataset mnist --distribution iid

# CIFAR-10数据集，Non-IID分布，结构化稀疏化
python main_sparsification.py --dataset cifar10 --distribution non-iid-dir \
    --alpha 0.5 --sparsification_mode structured --lambda_coef 2.0
```

### 批量实验 / Batch Experiments

```bash
# 运行基础实验套件
python run_experiments_sparsification.py --experiment basic

# 运行对比实验（不同稀疏化模式）
python run_experiments_sparsification.py --experiment comparison

# 运行完整实验套件
python run_experiments_sparsification.py --experiment full
```

## 📝 命令行参数 / Command Line Arguments

### 基础参数 / Basic Parameters

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--dataset` | str | cifar10 | 数据集: mnist, fashion-mnist, cifar10, cifar100 |
| `--num_clients` | int | 100 | 客户端数量 |
| `--num_rounds` | int | 50 | 通信轮次 |
| `--distribution` | str | non-iid-dir | 数据分布: iid, non-iid-dir |
| `--alpha` | float | 0.5 | Dirichlet分布参数 (用于non-iid-dir) |

### 稀疏化参数 / Sparsification Parameters

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--sparsification_mode` | str | magnitude | 稀疏化模式: magnitude, random, structured |
| `--lambda_coef` | float | 2.0 | 保留率计算的λ系数 |
| `--min_keep_ratio` | float | 0.1 | 最小保留率 |

### 训练参数 / Training Parameters

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--local_epochs` | int | 5 | 本地训练轮次 |
| `--batch_size` | int | 32 | 批次大小 |
| `--learning_rate` | float | 0.01 | 学习率 |
| `--standalone_epochs` | int | 20 | 独立训练轮次（用于基准） |

## 🔬 稀疏化模式 / Sparsification Modes

### 1. Magnitude-based (基于权重大小)
- 保留权重绝对值最大的参数
- 适合一般的神经网络模型
- 计算效率高

### 2. Random (随机稀疏化)
- 随机选择要保留的参数
- 作为基准对比方法
- 不考虑参数重要性

### 3. Structured (结构化稀疏化)
- 按整个滤波器/通道进行稀疏化
- 可以实现实际的加速效果
- 适合卷积神经网络

## 📊 评估指标 / Evaluation Metrics

### PCC (Pearson Correlation Coefficient)
- 评估独立训练与联邦学习性能的相关性
- 目标：提高PCC值（>0.6为良好）

### IPR (Incentivized Participation Rate)
- 激励参与率：受益客户端的比例
- 公式：IPR = (获得性能提升的客户端数) / 总客户端数
- 目标：IPR > 0.8

### 实际稀疏率统计
- 各等级客户端的平均保留率
- 稀疏化的实际效果
- 模型压缩比例

## 📁 项目结构 / Project Structure

```
.
├── main_sparsification.py              # 主程序
├── run_experiments_sparsification.py   # 实验运行脚本
├── config_updated.py                   # 配置文件
├── incentive/
│   ├── sparsification_distributor.py  # 稀疏化分发器（核心模块）
│   ├── membership.py                  # 会员系统
│   ├── time_slice.py                  # 时间片管理
│   └── points_calculator.py           # CGSV贡献度计算
├── federated/
│   ├── server_sparsification.py       # 联邦服务器（稀疏化版本）
│   └── client.py                      # 联邦客户端
├── outputs/
│   ├── results/                       # 实验结果JSON
│   └── figures/                       # 可视化图表
└── README_sparsification.md           # 本文档
```

## 🧪 实验示例 / Experiment Examples

### 示例1: 测试不同λ值的影响

```bash
# λ=1 (线性关系)
python main_sparsification.py --dataset cifar10 --lambda_coef 1.0

# λ=2 (凸函数，默认)
python main_sparsification.py --dataset cifar10 --lambda_coef 2.0

# λ=3 (更凸的函数，高贡献者优势更明显)
python main_sparsification.py --dataset cifar10 --lambda_coef 3.0
```

### 示例2: 对比不同数据分布

```bash
# IID分布
python main_sparsification.py --dataset cifar10 --distribution iid

# Non-IID (α=0.5, 中等异质性)
python main_sparsification.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5

# Non-IID (α=0.1, 高度异质性)
python main_sparsification.py --dataset cifar10 --distribution non-iid-dir --alpha 0.1
```

### 示例3: 大规模实验

```bash
python main_sparsification.py \
    --dataset cifar100 \
    --num_clients 200 \
    --num_rounds 150 \
    --distribution non-iid-dir \
    --alpha 0.5 \
    --sparsification_mode structured \
    --lambda_coef 2.5 \
    --local_epochs 5 \
    --standalone_epochs 30
```

## 📈 预期改进 / Expected Improvements

相比原始的UPSM方法，稀疏化方法预期带来以下改进：

1. **更高的PCC值**: 预期从0.4提升到0.6+
2. **更稳定的收敛**: 减少客户端之间的性能差异
3. **计算效率**: 稀疏模型减少客户端的计算负担
4. **通信效率**: 可以只传输非零参数的位置和值
5. **公平性提升**: 低贡献客户端也能获得基础模型功能

## 🔍 调试和优化 / Debugging and Optimization

### 如果PCC仍然较低：

1. **调整λ值**: 尝试1.5, 2.0, 2.5, 3.0
2. **修改最小保留率**: 尝试0.05, 0.1, 0.15, 0.2
3. **改变稀疏化模式**: 从magnitude改为structured
4. **调整会员等级比例**: 修改LEVEL_PERCENTILES
5. **增加训练轮次**: 确保模型充分收敛

### 监控指标：

```python
# 在实验过程中会打印以下关键信息：
- 每轮的平均准确率和稀疏化统计
- 各会员等级的分布和平均稀疏率
- CGSV贡献度的分布
- PCC和IPR的变化趋势
```

## 📚 参考文献 / References

1. 原始UPSM方法: "Unified Probabilistic Sampling Mechanism for Federated Learning"
2. 模型剪枝: "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
3. 联邦学习激励: "Incentive Mechanism Design for Federated Learning"
4. CGSV贡献度: "Cosine Gradient Shapley Value for Contribution Evaluation"

## 💡 创新点总结 / Innovation Summary

1. **方法论创新**: 从"加法"（选择性聚合）转向"减法"（差异化稀疏）
2. **双重控制**: 结合会员等级（离散）和贡献度（连续）进行精细控制
3. **实用性**: 稀疏化不仅差异化奖励，还带来实际的计算和通信优势
4. **公平性**: 保证所有客户端都能获得可用的模型（至少10%参数）

---

**注意 / Note**: 本代码是研究原型，实际部署时需要考虑安全性、隐私保护和系统鲁棒性等因素。

**作者 / Author**: Ziqian (Research on Federated Learning Incentive Mechanisms)