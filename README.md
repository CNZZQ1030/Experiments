# 联邦学习项目扩展使用指南
# Federated Learning Extended Project Usage Guide

## 📋 更新概述 / Update Overview

本次更新添加了以下功能：

### 新增数据集 / New Datasets
1. **CIFAR-100** - 100类图像分类
2. **SST** - Stanford Sentiment Treebank 情感分析

### 新增数据分布类型 / New Distribution Types
1. **iid** - 独立同分布 / IID
2. **non-iid-dir** - Dirichlet分布 (原non-iid) / Dirichlet distribution
3. **non-iid-size** - 数据量不平衡 / Imbalanced dataset size
4. **non-iid-class** - 类别数不平衡 / Imbalanced class number

---

## 🚀 快速开始 / Quick Start

### 基本命令格式 / Basic Command Format
```bash
python main.py --dataset <DATASET> --distribution <DISTRIBUTION> [OPTIONS]
```

---

## 📝 命令行参数详解 / Command Line Arguments

### 数据集参数 / Dataset Parameters
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--dataset` | str | mnist | 数据集: mnist, fashion-mnist, cifar10, cifar100, sst |
| `--num_clients` | int | 100 | 客户端数量 |

### 分布参数 / Distribution Parameters
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--distribution` | str | iid | 分布类型: iid, non-iid-dir, non-iid-size, non-iid-class |
| `--alpha` | float | 0.5 | Dirichlet参数 (用于non-iid-dir) |
| `--size_ratio` | float | 5.0 | 数据量不平衡比例 (用于non-iid-size) |
| `--min_classes` | int | 2 | 每客户端最少类别数 (用于non-iid-class) |
| `--max_classes` | int | 5 | 每客户端最多类别数 (用于non-iid-class) |

### 训练参数 / Training Parameters
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--num_rounds` | int | 50 | 通信轮次 |
| `--local_epochs` | int | 5 | 每轮本地训练轮次 |
| `--batch_size` | int | 32 | 批次大小 |
| `--learning_rate` | float | 0.01 | 学习率 |
| `--standalone_epochs` | int | 20 | 独立训练轮次 |

### 其他参数 / Other Parameters
| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--seed` | int | 42 | 随机种子 |
| `--device` | str | auto | 计算设备: auto, cpu, cuda |

---

## 📚 使用示例 / Usage Examples

### 1. IID分布实验 / IID Distribution Experiments

```bash
# MNIST with IID / MNIST IID分布
python main.py --dataset mnist --distribution iid --num_clients 100 --num_rounds 50

# CIFAR-10 with IID / CIFAR-10 IID分布
python main.py --dataset cifar10 --distribution iid --num_clients 100 --num_rounds 100

# CIFAR-100 with IID / CIFAR-100 IID分布
python main.py --dataset cifar100 --distribution iid --num_clients 100 --num_rounds 150

# SST with IID / SST IID分布
python main.py --dataset sst --distribution iid --num_clients 50 --num_rounds 30
```

### 2. Dirichlet Non-IID分布 / Dirichlet Non-IID Distribution

```bash
# MNIST with Dirichlet (α=0.1, 高度非独立同分布)
python main.py --dataset mnist --distribution non-iid-dir --alpha 0.1

# CIFAR-10 with Dirichlet (α=0.5, 中等非独立同分布)
python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5

# CIFAR-100 with Dirichlet (α=1.0, 轻度非独立同分布)
python main.py --dataset cifar100 --distribution non-iid-dir --alpha 1.0

# SST with Dirichlet
python main.py --dataset sst --distribution non-iid-dir --alpha 0.5
```

**Alpha参数说明 / Alpha Parameter Guide:**
- α < 0.1: 极端Non-IID (每个客户端几乎只有1-2个类别)
- α = 0.5: 中等Non-IID
- α = 1.0: 轻度Non-IID
- α > 10: 接近IID

### 3. 数据量不平衡分布 / Imbalanced Dataset Size

```bash
# MNIST with size ratio 5.0 (最大客户端数据量是最小的5倍)
python main.py --dataset mnist --distribution non-iid-size --size_ratio 5.0

# CIFAR-10 with size ratio 10.0
python main.py --dataset cifar10 --distribution non-iid-size --size_ratio 10.0

# CIFAR-100 with size ratio 8.0
python main.py --dataset cifar100 --distribution non-iid-size --size_ratio 8.0
```

### 4. 类别数不平衡分布 / Imbalanced Class Number

```bash
# MNIST: 每客户端1-3个类别 (极端不平衡)
python main.py --dataset mnist --distribution non-iid-class --min_classes 1 --max_classes 3

# CIFAR-10: 每客户端2-5个类别
python main.py --dataset cifar10 --distribution non-iid-class --min_classes 2 --max_classes 5

# CIFAR-100: 每客户端5-20个类别
python main.py --dataset cifar100 --distribution non-iid-class --min_classes 5 --max_classes 20
```

### 5. 完整配置示例 / Full Configuration Examples

```bash
# 完整CIFAR-10 Dirichlet实验
python main.py \
    --dataset cifar10 \
    --distribution non-iid-dir \
    --alpha 0.5 \
    --num_clients 100 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --standalone_epochs 20 \
    --seed 42

# 完整CIFAR-100类别不平衡实验
python main.py \
    --dataset cifar100 \
    --distribution non-iid-class \
    --min_classes 10 \
    --max_classes 30 \
    --num_clients 50 \
    --num_rounds 150 \
    --local_epochs 3 \
    --batch_size 64 \
    --learning_rate 0.005 \
    --standalone_epochs 30

# SST文本分类实验
python main.py \
    --dataset sst \
    --distribution non-iid-dir \
    --alpha 0.5 \
    --num_clients 30 \
    --num_rounds 50 \
    --local_epochs 5 \
    --batch_size 32
```

---

## 📊 分布类型对比 / Distribution Type Comparison

| 分布类型 | 数据量 | 类别分布 | 适用场景 |
|---------|--------|---------|---------|
| iid | 均匀 | 均匀 | 基准测试 |
| non-iid-dir | 不均匀 | 不均匀 | 模拟真实场景的标签偏斜 |
| non-iid-size | 不均匀 | 均匀 | 模拟设备存储容量差异 |
| non-iid-class | 均匀 | 不均匀 | 模拟专业化设备 |

---

## 🔧 代码修改说明 / Code Modification Details

### 修改的文件 / Modified Files

1. **config.py**
   - 添加新分布类型支持
   - 添加SST数据集配置
   - 添加不平衡分布参数

2. **datasets/data_loader.py**
   - 添加SSTDataset类
   - 添加`_create_size_imbalanced_splits()`方法
   - 添加`_create_class_imbalanced_splits()`方法
   - 重构`_create_dirichlet_splits()`方法

3. **models/cnn_model.py**
   - 添加TextCNN模型
   - 添加TextLSTM模型
   - 更新ModelFactory

4. **main.py**
   - 添加新命令行参数
   - 更新组件初始化逻辑
   - 添加使用示例

---

## 📈 输出文件 / Output Files

运行实验后，结果将保存在以下位置：

```
outputs/
├── results/           # 实验结果JSON
├── figures/           # 可视化图表
└── logs/             # 日志文件
```

---

## ⚠️ 注意事项 / Notes

1. **CIFAR-100** 需要更多训练轮次(建议150+)以达到收敛
2. **SST** 数据集首次运行时会生成模拟数据，实际使用时可替换为真实数据
3. **non-iid-class** 分布中，`max_classes`不能超过数据集的类别总数
4. 使用GPU时建议增大`batch_size`以提高效率

---

## 📞 故障排除 / Troubleshooting

**Q: 内存不足 / Out of Memory**
```bash
# 减小batch_size和num_clients
python main.py --dataset cifar100 --batch_size 16 --num_clients 50
```

**Q: 训练太慢 / Training too slow**
```bash
# 减少local_epochs或num_rounds
python main.py --dataset cifar10 --local_epochs 3 --num_rounds 30
```

**Q: SST数据集找不到 / SST dataset not found**
```
首次运行时会自动生成模拟数据，保存在 ./data/sst/ 目录
```

---

## 🎯 推荐实验配置 / Recommended Configurations

### 快速测试 / Quick Test
```bash
python main.py --dataset mnist --distribution iid --num_clients 10 --num_rounds 10
```

### 标准IID基准 / Standard IID Baseline
```bash
python main.py --dataset cifar10 --distribution iid --num_clients 100 --num_rounds 100
```

### Non-IID性能测试 / Non-IID Performance Test
```bash
python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5 --num_clients 100 --num_rounds 100
```

### 极端Non-IID测试 / Extreme Non-IID Test
```bash
python main.py --dataset cifar10 --distribution non-iid-class --min_classes 1 --max_classes 2 --num_clients 100 --num_rounds 150
```