"""
config.py - 基于稀疏化的差异化模型配置（重构版）
Configuration for Sparsification-based Differentiated Model Distribution (Refactored)

新增：基于贡献度的加权聚合配置
Added: Contribution-Aware Aggregation Configuration
"""

import torch
import os
from datetime import datetime

# =====================================
# 基础配置 / Basic Configuration
# =====================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
SEED = 42

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# =====================================
# 联邦学习配置 / Federated Learning Configuration
# =====================================

class FederatedConfig:
    """联邦学习配置 / Federated Learning Configuration"""
    
    NUM_CLIENTS = 100
    NUM_ROUNDS = 50
    STANDALONE_EPOCHS = 20
    LOCAL_EPOCHS = 1
    LOCAL_BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    DISTRIBUTION_TYPE = "non-iid-dir"
    NON_IID_ALPHA = 0.5
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

# =====================================
# 激励机制配置 - 稀疏化版本 / Incentive Configuration - Sparsification Version
# =====================================

class IncentiveConfig:
    """激励机制参数配置 - 基于稀疏化的差异化
    Incentive Mechanism - Sparsification-based Differentiation"""
    
    # ===== CGSV配置 / CGSV Configuration =====
    CGSV_EPSILON = 1e-10
    
    # ===== 时间片配置 / Time Slice Configuration =====
    TIME_SLICE_TYPE = "rounds"
    ROUNDS_PER_SLICE = 5  # τ: 每个时间片包含的轮次数
    POINTS_VALIDITY_SLICES = 2  # W: 积分有效期（时间片数）
    
    # ===== 会员等级配置（基于相对排名）/ Membership Level Configuration =====
    # Diamond 10%, Gold 30%, Silver 40%, Bronze 20%
    LEVEL_PERCENTILES = {
        'diamond': 0.90,  # Top 10%
        'gold': 0.60,     # Top 40%
        'silver': 0.20,   # Top 80%
        'bronze': 0.00    # Remaining 20%
    }
    
    # ===== 稀疏化配置 / Sparsification Configuration =====
    # 基础稀疏率范围 / Base sparsity rate ranges
    # sparsity_rate: 被置零的参数比例 / Proportion of parameters set to zero
    LEVEL_SPARSITY_RANGES = {
        'diamond': (0.0, 0.1),    # 稀疏率 0-10% (保留90-100%参数)
        'gold': (0.1, 0.3),       # 稀疏率 10-30% (保留70-90%参数)
        'silver': (0.3, 0.6),     # 稀疏率 30-60% (保留40-70%参数)
        'bronze': (0.6, 0.95)     # 稀疏率 60-95% (保留5-40%参数)
    }
    
    # 保留率计算参数 / Keep ratio calculation parameters
    MIN_KEEP_RATIO = 0.1  # 最低保留率 10% / Minimum keep ratio
    MAX_KEEP_RATIO = 1.0  # 最高保留率 100% / Maximum keep ratio
    
    # 调节系数 λ (控制曲线形状) / Adjustment coefficient λ
    # λ > 1: 凸函数，高贡献者优势更明显 / Convex function, advantage for high contributors
    # λ = 1: 线性关系 / Linear relationship
    # λ < 1: 凹函数，更均衡 / Concave function, more balanced
    LAMBDA = 2.0
    
    # 稀疏化模式 / Sparsification mode
    # "magnitude": 基于权重大小的稀疏化 / Magnitude-based pruning
    # "random": 随机稀疏化 / Random pruning
    # "structured": 结构化稀疏化（按通道/层）/ Structured pruning
    SPARSIFICATION_MODE = "magnitude"
    
    # 结构化稀疏化粒度 / Structured sparsification granularity
    # "filter": 按卷积核稀疏化 / Filter-wise pruning
    # "channel": 按通道稀疏化 / Channel-wise pruning
    # "layer": 按层稀疏化 / Layer-wise pruning
    STRUCTURED_GRANULARITY = "filter"
    
    # 是否使用渐进式稀疏化 / Progressive sparsification
    PROGRESSIVE_SPARSIFICATION = False
    SPARSIFICATION_WARMUP_ROUNDS = 10
    
    # ===== 新增：基于贡献度的加权聚合配置 =====
    # ===== NEW: Contribution-Aware Aggregation Configuration =====
    
    # 聚合方式 / Aggregation method
    # "fedavg": 传统FedAvg（基于样本数量）/ Traditional FedAvg (sample-based)
    # "contribution": 基于贡献度的加权聚合 / Contribution-aware aggregation
    AGGREGATION_METHOD = "contribution"
    
    # Scale参数：控制Softmax的温度（区分度）
    # Scale parameter: Controls Softmax temperature (discrimination)
    # 较大值(>10): 趋向"赢家通吃" / Larger values: "winner-takes-all"
    # 较小值(~1): 更均衡的权重分配 / Smaller values: more balanced
    # 建议范围: 1.0 - 10.0
    AGGREGATION_SCALE = 5.0
    
    # Scale模式 / Scale mode
    # "fixed": 使用固定的AGGREGATION_SCALE值 / Use fixed AGGREGATION_SCALE
    # "dynamic": 动态调整为 1/std(contributions) / Dynamic: 1/std(contributions)
    AGGREGATION_SCALE_MODE = "fixed"
    
    # 混合权重：结合样本数量和贡献度 / Hybrid weight: combine sample count and contribution
    # final_weight = SAMPLE_WEIGHT_RATIO * sample_weight + (1-SAMPLE_WEIGHT_RATIO) * contribution_weight
    # 0.0: 纯贡献度加权 / Pure contribution weighting
    # 1.0: 纯样本数量加权（FedAvg）/ Pure sample weighting (FedAvg)
    # 0.5: 混合 / Hybrid
    SAMPLE_WEIGHT_RATIO = 0.0  # 默认使用纯贡献度加权

# =====================================
# 数据集配置 / Dataset Configuration
# =====================================

class DatasetConfig:
    """数据集参数配置 / Dataset Parameters"""
    
    AVAILABLE_DATASETS = ["mnist", "fashion-mnist", "cifar10", "cifar100"]
    DATASET_NAME = "cifar10"
    DATA_ROOT = "./data"
    
    NORMALIZE_MEAN = {
        "mnist": (0.1307,),
        "fashion-mnist": (0.2860,),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408)
    }
    
    NORMALIZE_STD = {
        "mnist": (0.3081,),
        "fashion-mnist": (0.3530,),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761)
    }
    
    INPUT_SHAPE = {
        "mnist": (1, 28, 28),
        "fashion-mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32)
    }
    
    NUM_CLASSES = {
        "mnist": 10,
        "fashion-mnist": 10,
        "cifar10": 10,
        "cifar100": 100
    }

# =====================================
# 模型配置 / Model Configuration
# =====================================

class ModelConfig:
    """模型参数配置 / Model Parameters"""
    
    CNN_CHANNELS = [32, 64]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.5
    
    # 稀疏化相关配置 / Sparsification-related configuration
    # 不同层的稀疏化敏感度权重 / Layer sensitivity weights for sparsification
    LAYER_IMPORTANCE_WEIGHTS = {
        'conv': 1.0,   # 卷积层权重 / Convolution layer weight
        'fc': 0.8,     # 全连接层权重 / Fully connected layer weight
        'bn': 0.0,     # BN层不稀疏化 / Don't sparsify BN layers
        'first': 0.5,  # 第一层权重 / First layer weight
        'last': 0.5    # 最后一层权重 / Last layer weight
    }
    
    # 新增：模型选择配置 / NEW: Model selection configuration
    # CIFAR-100使用更高级的模型 / Use advanced model for CIFAR-100
    MODEL_TYPE = {
        "mnist": "simple_cnn",
        "fashion-mnist": "simple_cnn",
        "cifar10": "cifar_cnn",
        "cifar100": "resnet18"  # 使用ResNet18 / Use ResNet18
    }

# =====================================
# 实验配置 / Experiment Configuration
# =====================================

class ExperimentConfig:
    """实验参数配置 / Experiment Parameters"""
    
    EXPERIMENT_NAME = f"FL_Sparsification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    EVAL_FREQUENCY = 5
    SAVE_FREQUENCY = 10
    LOG_LEVEL = "INFO"
    
    BASE_OUTPUT_DIR = "outputs"
    CHECKPOINT_DIR = f"{BASE_OUTPUT_DIR}/checkpoints"
    LOG_DIR = f"{BASE_OUTPUT_DIR}/logs"
    PLOTS_DIR = f"{BASE_OUTPUT_DIR}/plots"

# =====================================
# 创建必要的目录 / Create necessary directories
# =====================================

def setup_directories():
    """创建输出目录 / Create output directories"""
    directories = [
        ExperimentConfig.CHECKPOINT_DIR,
        ExperimentConfig.LOG_DIR,
        ExperimentConfig.PLOTS_DIR
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

setup_directories()