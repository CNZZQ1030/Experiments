"""
config.py - 层级约束动态梯度奖励配置（扩展文本数据集版本 - 修正版）
Configuration for Tier-Constrained Dynamic Gradient Reward (Extended with Text Datasets - Fixed)

基于NeurIPS 2021论文"Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"
Based on NeurIPS 2021 paper "Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"

修正说明 / Fix Notes:
- 移除了 TEXT_CONFIG 中的 vocab_size，避免参数冲突
- vocab_size 现在由数据加载器动态确定
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
# 激励机制配置 - 层级约束动态梯度奖励版本
# Incentive Configuration - Tier-Constrained Dynamic Gradient Reward Version
# =====================================

class IncentiveConfig:
    """
    激励机制参数配置 - 层级约束动态梯度奖励
    Incentive Mechanism - Tier-Constrained Dynamic Gradient Reward
    
    核心思想 / Core Concept:
    - 不给每个等级一个"固定"的稀疏率
    - 将等级作为稀疏率的上下界（Bounds）
    - 在界限内根据CGSV分数进行连续且敏感的映射
    
    - Don't give each tier a "fixed" sparsification rate
    - Use tiers as bounds for sparsification rates
    - Map continuously and sensitively within bounds based on CGSV scores
    """
    
    # ===== CGSV配置 / CGSV Configuration =====
    CGSV_EPSILON = 1e-10
    
    # ===== 时间片配置 / Time Slice Configuration =====
    TIME_SLICE_TYPE = "rounds"
    ROUNDS_PER_SLICE = 5  # τ: 每个时间片包含的轮次数 / Rounds per time slice
    POINTS_VALIDITY_SLICES = 2  # W: 积分有效期（时间片数）/ Points validity period
    
    # ===== 会员等级配置（基于相对排名）/ Membership Level Configuration =====
    # 使用三级制：Gold, Silver, Bronze
    # Using three-tier system: Gold, Silver, Bronze
    # percentile值表示该等级需要的最低排名百分位
    # Percentile value indicates the minimum ranking percentile for that level
    LEVEL_PERCENTILES = {
        'gold': 0.80,     # Top 20% -> percentile >= 0.80
        'silver': 0.50,   # Next 30% -> 0.50 <= percentile < 0.80
        'bronze': 0.00    # Bottom 50% -> percentile < 0.50
    }
    
    # ===== 核心改进：层级约束的稀疏率范围 =====
    # ===== Core Improvement: Tier-Constrained Sparsification Rate Ranges =====
    # 
    # 关键改变 / Key Changes:
    # 1. Bronze等级的下限大幅降低到0.1（只保留10%参数）
    # 2. 使用保留率(keep_ratio)而非稀疏率(sparsity_rate)
    # 3. 组内使用线性插值而非固定值
    #
    # keep_ratio范围：[lower_bound, upper_bound]
    # keep_ratio range: [lower_bound, upper_bound]
    TIER_KEEP_RATIO_RANGES = {
        'gold': (0.80, 1.0),    # 高贡献：保留80%-100%参数 / High contribution: keep 80%-100%
        'silver': (0.50, 0.80), # 中贡献：保留50%-80%参数 / Medium contribution: keep 50%-80%
        'bronze': (0.10, 0.50)  # 低贡献：保留10%-50%参数 / Low contribution: keep 10%-50%
    }
    
    # 备选配置：更激进的差异化（用于实验）
    # Alternative config: More aggressive differentiation (for experiments)
    TIER_KEEP_RATIO_RANGES_AGGRESSIVE = {
        'gold': (0.90, 1.0),    # 保留90%-100% / Keep 90%-100%
        'silver': (0.60, 0.90), # 保留60%-90% / Keep 60%-90%
        'bronze': (0.10, 0.60)  # 保留10%-60% / Keep 10%-60%
    }
    
    # 备选配置：更温和的差异化
    # Alternative config: More moderate differentiation
    TIER_KEEP_RATIO_RANGES_MODERATE = {
        'gold': (0.70, 1.0),    # 保留70%-100% / Keep 70%-100%
        'silver': (0.40, 0.70), # 保留40%-70% / Keep 40%-70%
        'bronze': (0.20, 0.40)  # 保留20%-40% / Keep 20%-40%
    }
    
    # ===== 全局保留率限制 / Global Keep Ratio Limits =====
    MIN_KEEP_RATIO = 0.05  # 绝对最低保留率5% / Absolute minimum keep ratio
    MAX_KEEP_RATIO = 1.0   # 最高保留率100% / Maximum keep ratio
    
    # ===== 稀疏化模式 / Sparsification Mode =====
    # "magnitude": 基于权重大小的稀疏化（推荐）/ Magnitude-based pruning (recommended)
    # "random": 随机稀疏化 / Random pruning
    # "structured": 结构化稀疏化（按通道/层）/ Structured pruning
    SPARSIFICATION_MODE = "magnitude"
    
    # ===== 组内插值配置 / Intra-Tier Interpolation Configuration =====
    # 插值方法 / Interpolation method
    # "linear": 线性插值 / Linear interpolation
    # "power": 幂律插值（可调节曲线形状）/ Power-law interpolation
    INTERPOLATION_METHOD = "linear"
    
    # 幂律插值的指数（仅当INTERPOLATION_METHOD="power"时有效）
    # Exponent for power-law interpolation (only effective when INTERPOLATION_METHOD="power")
    # λ > 1: 凸函数，高贡献者优势更明显 / Convex function, advantage for high contributors
    # λ = 1: 线性关系 / Linear relationship
    # λ < 1: 凹函数，更均衡 / Concave function, more balanced
    INTERPOLATION_LAMBDA = 1.0
    
    # ===== 移动平均配置（参考论文公式4）=====
    # ===== Moving Average Configuration (Reference: Paper Equation 4) =====
    # r_{i,t} = α * r_{i,t-1} + (1-α) * ψ_{i,t}
    MOVING_AVERAGE_ALPHA = 0.95
    
    # ===== 聚合方式配置 =====
    # ===== Aggregation Method Configuration =====
    # "fedavg": 传统FedAvg（基于样本数量）/ Traditional FedAvg (sample-based)
    # "contribution": 基于贡献度的加权聚合 / Contribution-aware aggregation
    AGGREGATION_METHOD = "contribution"
    
    # Scale参数：控制Softmax的温度（区分度）
    # Scale parameter: Controls Softmax temperature (discrimination)
    AGGREGATION_SCALE = 5.0
    
    # ===== 调试和日志配置 / Debug and Logging Configuration =====
    VERBOSE_SPARSIFICATION = True  # 是否打印详细的稀疏化信息 / Print detailed sparsification info

# =====================================
# 数据集配置（扩展版本）/ Dataset Configuration (Extended)
# =====================================

class DatasetConfig:
    """
    数据集参数配置 - 扩展版本
    Dataset Parameters - Extended Version
    
    新增支持 / New Support:
    - mr: Movie Review 电影评论情感分析
    - sst: Stanford Sentiment Treebank 斯坦福情感树库
    """
    
    # 可用数据集列表 / Available datasets list
    AVAILABLE_DATASETS = [
        "mnist", "fashion-mnist", "cifar10", "cifar100",  # 图像数据集 / Image datasets
        "mr", "sst"  # 文本数据集 / Text datasets (新增 / New)
    ]
    
    DATASET_NAME = "cifar10"
    DATA_ROOT = "./data"
    
    # ===== 图像数据集归一化参数 / Image Dataset Normalization =====
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
    
    # ===== 输入形状 / Input Shape =====
    INPUT_SHAPE = {
        "mnist": (1, 28, 28),
        "fashion-mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "mr": None,  # 文本数据，形状由模型决定 / Text data, shape determined by model
        "sst": None  # 文本数据 / Text data
    }
    
    # ===== 类别数量 / Number of Classes =====
    NUM_CLASSES = {
        "mnist": 10,
        "fashion-mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "mr": 2,  # 二分类：正面/负面 / Binary: positive/negative
        "sst": 2  # 二分类：正面/负面 / Binary: positive/negative
    }
    
    # ===== 文本数据集特定参数 / Text Dataset Specific Parameters =====
    # 重要：vocab_size 不在此配置，由数据加载器动态确定
    # Important: vocab_size is NOT configured here, determined dynamically by data loader
    TEXT_CONFIG = {
        "mr": {
            "embedding_dim": 128,
            "max_seq_length": 200,
            "filter_sizes": [3, 4, 5],  # TextCNN卷积核大小 / TextCNN filter sizes
            "num_filters": 100,  # 每个卷积核的数量 / Number of filters per size
            "dropout": 0.5
        },
        "sst": {
            "embedding_dim": 128,
            "max_seq_length": 200,
            "filter_sizes": [3, 4, 5],
            "num_filters": 100,
            "dropout": 0.5
        }
    }
    
    @staticmethod
    def is_text_dataset(dataset_name: str) -> bool:
        """
        判断是否为文本数据集 / Check if it's a text dataset
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            
        Returns:
            bool: 是否为文本数据集 / Whether it's a text dataset
        """
        return dataset_name.lower() in ["mr", "sst"]
    
    @staticmethod
    def get_text_config(dataset_name: str) -> dict:
        """
        获取文本数据集配置 / Get text dataset configuration
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            
        Returns:
            dict: 文本数据集配置 / Text dataset configuration
        """
        if dataset_name.lower() in DatasetConfig.TEXT_CONFIG:
            return DatasetConfig.TEXT_CONFIG[dataset_name.lower()].copy()  # 返回副本避免修改原配置
        return {}

# =====================================
# 模型配置（扩展版本）/ Model Configuration (Extended)
# =====================================

class ModelConfig:
    """
    模型参数配置 - 扩展版本
    Model Parameters - Extended Version
    
    支持图像和文本模型 / Support both image and text models
    """
    
    # ===== CNN配置（图像模型）/ CNN Configuration (Image Models) =====
    CNN_CHANNELS = [32, 64]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.5
    
    # 不同层的稀疏化敏感度权重（可选）
    # Layer sensitivity weights for sparsification (optional)
    LAYER_IMPORTANCE_WEIGHTS = {
        'conv': 1.0,   # 卷积层权重 / Convolution layer weight
        'fc': 0.8,     # 全连接层权重 / Fully connected layer weight
        'bn': 0.0,     # BN层不稀疏化 / Don't sparsify BN layers
        'first': 0.5,  # 第一层权重 / First layer weight
        'last': 0.5    # 最后一层权重 / Last layer weight
    }
    
    # ===== 模型选择配置 / Model Selection Configuration =====
    MODEL_TYPE = {
        # 图像模型 / Image models
        "mnist": "simple_cnn",
        "fashion-mnist": "simple_cnn",
        "cifar10": "cifar_cnn",
        "cifar100": "resnet18",
        
        # 文本模型 / Text models (新增 / New)
        "mr": "textcnn",
        "sst": "textcnn"
    }

# =====================================
# 实验配置 / Experiment Configuration
# =====================================

class ExperimentConfig:
    """实验参数配置 / Experiment Parameters"""
    
    EXPERIMENT_NAME = f"FL_TierConstrainedGradient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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