"""
config.py (Updated for UPSM Implementation)
配置文件 - UPSM统一概率采样机制 / Configuration File - UPSM Implementation
"""

import torch
import os
from datetime import datetime

# =====================================
# 基础配置 / Basic Configuration
# =====================================

# 设备配置 / Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# 随机种子 / Random seed
SEED = 42

# 输出路径 / Output paths
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# =====================================
# 联邦学习配置 / Federated Learning Configuration
# =====================================

class FederatedConfig:
    """联邦学习配置 / Federated Learning Configuration"""
    
    # 基本参数 / Basic parameters
    NUM_CLIENTS = 10
    NUM_ROUNDS = 50

    # 本地训练参数 / Local training parameters
    STANDALONE_EPOCHS = 10
    LOCAL_EPOCHS = 1
    LOCAL_BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # 数据分布参数 / Data distribution parameters
    DISTRIBUTION_TYPE = "iid"
    NON_IID_ALPHA = 0.5
    
    # 不平衡分布参数 / Imbalanced distribution parameters
    SIZE_IMBALANCE_RATIO = 5.0
    MIN_CLASSES_PER_CLIENT = 2
    MAX_CLASSES_PER_CLIENT = 5
    
    # 优化器参数 / Optimizer parameters
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

# =====================================
# 激励机制配置 / Incentive Mechanism Configuration
# =====================================

class IncentiveConfig:
    """激励机制参数配置 / Incentive Mechanism Parameters"""
    
    # ===== CGSV 相关配置 / CGSV Related Configuration =====
    CGSV_EPSILON = 1e-10
    
    # ===== 时间片配置 / Time Slice Configuration =====
    TIME_SLICE_TYPE = "rounds"
    ROUNDS_PER_SLICE = 5  # τ: 每个时间片包含的轮次数 / Rounds per time slice
    POINTS_VALIDITY_SLICES = 2  # W: 积分有效期（时间片数）/ Validity window size
    
    # ===== 会员等级配置（基于相对排名）/ Membership Level Configuration (Relative Ranking) =====
    # 按照PDF文档：Diamond 10%, Gold 30%, Silver 40%, Bronze 20%
    # According to PDF: Diamond 10%, Gold 30%, Silver 40%, Bronze 20%
    LEVEL_PERCENTILES = {
        'diamond': 0.90,  # Top 10% (前10%)
        'gold': 0.60,     # Top 40% (前40%, 即Gold占30%)
        'silver': 0.20,   # Top 80% (前80%, 即Silver占40%)
        'bronze': 0.00    # All remaining (剩余20%)
    }
    
    # 等级权益倍数（保留用于其他用途）/ Level benefit multipliers
    LEVEL_MULTIPLIERS = {
        'bronze': 1.0,
        'silver': 1.2,
        'gold': 1.5,
        'diamond': 2.0
    }
    
    # ===== UPSM配置 / UPSM Configuration =====
    # 数量控制：信息访问率 / Quantity Control: Information Access Ratio
    # ρ_L values from PDF Table 1
    LEVEL_ACCESS_RATIOS = {
        'diamond': 1.0,   # Full Access 全量访问
        'gold': 0.8,      # High Access 高访问量
        'silver': 0.5,    # Medium Access 中等访问量
        'bronze': 0.2     # Low Access 低访问量
    }
    
    # 质量控制：选择偏差系数 / Quality Control: Selection Bias Coefficient
    # β_L values from PDF Table 2
    LEVEL_SELECTION_BIAS = {
        'diamond': 10.0,  # Deterministic Exploitation 确定性择优
        'gold': 3.0,      # Probabilistic Exploitation 概率性择优
        'silver': 1.0,    # Weak Preference 弱偏好
        'bronze': 0.0     # Uniform Random 纯随机
    }
    
    # 最小可访问更新数 / Minimum accessible updates
    MIN_ACCESSIBLE_UPDATES = 1

# =====================================
# 数据集配置 / Dataset Configuration
# =====================================

class DatasetConfig:
    """数据集参数配置 / Dataset Parameters"""
    
    AVAILABLE_DATASETS = ["mnist", "fashion-mnist", "cifar10", "cifar100", "sst"]
    
    AVAILABLE_DISTRIBUTIONS = [
        "iid",
        "non-iid-dir",
        "non-iid-size",
        "non-iid-class"
    ]
    
    DATASET_NAME = "cifar10"
    DATA_ROOT = "./data"
    
    NORMALIZE_MEAN = {
        "mnist": (0.1307,),
        "fashion-mnist": (0.2860,),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "sst": None,
    }
    
    NORMALIZE_STD = {
        "mnist": (0.3081,),
        "fashion-mnist": (0.3530,),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "sst": None,
    }
    
    INPUT_SHAPE = {
        "mnist": (1, 28, 28),
        "fashion-mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "sst": (1, 200),
    }
    
    NUM_CLASSES = {
        "mnist": 10,
        "fashion-mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "sst": 2,
    }
    
    SST_MAX_SEQ_LENGTH = 200
    SST_VOCAB_SIZE = 20000
    SST_EMBEDDING_DIM = 128

# =====================================
# 模型配置 / Model Configuration
# =====================================

class ModelConfig:
    """模型参数配置 / Model Parameters"""
    
    CNN_CHANNELS = [32, 64]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.5
    
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    EMBEDDING_DIM = 128
    
    TEXTCNN_FILTER_SIZES = [3, 4, 5]
    TEXTCNN_NUM_FILTERS = 100

# =====================================
# 实验配置 / Experiment Configuration
# =====================================

class ExperimentConfig:
    """实验参数配置 / Experiment Parameters"""
    
    EXPERIMENT_NAME = f"FL_UPSM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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