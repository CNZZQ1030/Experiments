"""
config.py (Updated with Extended Datasets and Distributions)
配置文件 - 扩展数据集和分布类型 / Configuration File - Extended Datasets and Distributions
包含所有实验参数和系统设置
Contains all experimental parameters and system settings
"""

import torch
import os
from datetime import datetime

# =====================================
# 基础配置 / Basic Configuration
# =====================================

# 设备配置 / Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # 数据加载线程数 / Number of data loading threads

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
    NUM_CLIENTS = 10  # 客户端总数 / Total number of clients
    NUM_ROUNDS = 6  # 联邦学习总轮次（通信轮数） / FL training rounds (communication rounds)

    # 本地训练参数 / Local training parameters
    STANDALONE_EPOCHS = 10  # 独立训练轮次 / Standalone training epochs
    LOCAL_EPOCHS = 1  # 联邦学习每轮本地训练轮次 / Local training epochs per round
    LOCAL_BATCH_SIZE = 32  # 本地批次大小 / Local batch size
    LEARNING_RATE = 0.01  # 学习率 / Learning rate
    
    # 数据分布参数 / Data distribution parameters
    # 可选值 / Options: "iid", "non-iid-dir", "non-iid-size", "non-iid-class"
    DISTRIBUTION_TYPE = "iid"  # 数据分布类型 / Data distribution type
    NON_IID_ALPHA = 0.5  # Non-IID Dirichlet分布参数 / Dirichlet parameter for Non-IID
    
    # 不平衡分布参数 / Imbalanced distribution parameters
    # 用于 non-iid-size 分布 / For non-iid-size distribution
    SIZE_IMBALANCE_RATIO = 5.0  # 最大/最小客户端数据量比例 / Max/min client data ratio
    
    # 用于 non-iid-class 分布 / For non-iid-class distribution
    MIN_CLASSES_PER_CLIENT = 2  # 每个客户端最少类别数 / Minimum classes per client
    MAX_CLASSES_PER_CLIENT = 5  # 每个客户端最多类别数 / Maximum classes per client
    
    # 优化器参数 / Optimizer parameters
    MOMENTUM = 0.9  # SGD动量 / SGD momentum
    WEIGHT_DECAY = 1e-4  # 权重衰减 / Weight decay
    
    # 注意：在差异化模型分发机制下，所有客户端都参与训练
    # Note: Under differentiated model distribution, all clients participate

# =====================================
# 激励机制配置 / Incentive Mechanism Configuration
# =====================================

class IncentiveConfig:
    """激励机制参数配置 / Incentive Mechanism Parameters"""
    
    # ===== CGSV 相关配置 / CGSV Related Configuration =====
    CGSV_EPSILON = 1e-10  # CGSV余弦相似度计算的epsilon值 / Epsilon for cosine similarity
    
    # ===== 时间片配置 / Time Slice Configuration =====
    TIME_SLICE_TYPE = "rounds"  # 时间片类型 / Time slice type
    ROUNDS_PER_SLICE = 5  # 每个时间片包含的轮次数 / Rounds per time slice
    POINTS_VALIDITY_SLICES = 2  # 积分有效期（时间片数） / Points validity period
    DAYS_PER_SLICE = 3  # 基于天数的时间片长度 / Days per time slice
    
    # 动态时间片参数 / Dynamic time slice parameters
    ACTIVITY_THRESHOLD = 0.5  # 活跃度阈值 / Activity threshold
    BASE_SLICE_LENGTH = 10  # 基础时间片长度 / Base slice length
    
    # ===== 会员等级阈值 / Membership Level Thresholds =====
    LEVEL_THRESHOLDS = {
        'bronze': 0,
        'silver': 2000,
        'gold': 6000,
        'diamond': 15000
    }
    
    # 等级权益倍数 / Level benefit multipliers
    LEVEL_MULTIPLIERS = {
        'bronze': 1.0,
        'silver': 1.2,
        'gold': 1.5,
        'diamond': 2.0
    }
    
    # ===== 差异化模型奖励配置 / Differentiated Model Rewards Configuration =====
    ENABLE_TIERED_REWARDS = True
    
    LEVEL_ACCESS_RATIOS = {
        'diamond': 1.0,
        'gold': 0.8,
        'silver': 0.6,
        'bronze': 0.4
    }
    
    MIN_ACCESSIBLE_UPDATES = 1
    
    # CRC参数 / CRC parameters
    CRC_WINDOW_SIZE = 10
    CRC_MIN_SAMPLES = 3
    CRC_START_ROUND = 1

# =====================================
# 数据集配置 / Dataset Configuration
# =====================================

class DatasetConfig:
    """数据集参数配置 / Dataset Parameters"""
    
    # 支持的数据集 / Supported datasets
    # 更新：添加sst数据集 / Updated: Added sst dataset
    AVAILABLE_DATASETS = ["mnist", "fashion-mnist", "cifar10", "cifar100", "sst"]
    
    # 支持的分布类型 / Supported distribution types
    AVAILABLE_DISTRIBUTIONS = [
        "iid",           # 独立同分布 / Independent and identically distributed
        "non-iid-dir",   # Dirichlet分布 / Dirichlet distribution
        "non-iid-size",  # 数据量不平衡 / Imbalanced dataset size
        "non-iid-class"  # 类别数不平衡 / Imbalanced class number
    ]
    
    # 当前使用的数据集 / Current dataset
    DATASET_NAME = "cifar10"
    
    # 数据集路径 / Dataset path
    DATA_ROOT = "./data"
    
    # 数据预处理 / Data preprocessing
    NORMALIZE_MEAN = {
        "mnist": (0.1307,),
        "fashion-mnist": (0.2860,),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "sst": None,  # 文本数据无需归一化 / Text data doesn't need normalization
    }
    
    NORMALIZE_STD = {
        "mnist": (0.3081,),
        "fashion-mnist": (0.3530,),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "sst": None,
    }
    
    # 输入维度 / Input dimensions
    INPUT_SHAPE = {
        "mnist": (1, 28, 28),
        "fashion-mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
        "sst": (1, 200),  # (channels, max_seq_length)
    }
    
    # 类别数 / Number of classes
    NUM_CLASSES = {
        "mnist": 10,
        "fashion-mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "sst": 2,  # 二分类情感分析 / Binary sentiment classification
    }
    
    # SST特定配置 / SST specific configuration
    SST_MAX_SEQ_LENGTH = 200  # 最大序列长度 / Maximum sequence length
    SST_VOCAB_SIZE = 20000    # 词汇表大小 / Vocabulary size
    SST_EMBEDDING_DIM = 128   # 嵌入维度 / Embedding dimension

# =====================================
# 模型配置 / Model Configuration
# =====================================

class ModelConfig:
    """模型参数配置 / Model Parameters"""
    
    # CNN模型配置 / CNN model configuration
    CNN_CHANNELS = [32, 64]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.5
    
    # LSTM模型配置（用于SST数据集）/ LSTM model configuration (for SST dataset)
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    EMBEDDING_DIM = 128
    
    # TextCNN配置（用于SST数据集）/ TextCNN configuration (for SST dataset)
    TEXTCNN_FILTER_SIZES = [3, 4, 5]  # 卷积核大小 / Filter sizes
    TEXTCNN_NUM_FILTERS = 100         # 每种大小的卷积核数量 / Number of filters per size

# =====================================
# 实验配置 / Experiment Configuration
# =====================================

class ExperimentConfig:
    """实验参数配置 / Experiment Parameters"""
    
    EXPERIMENT_NAME = f"FL_Incentive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    EVAL_FREQUENCY = 5
    SAVE_FREQUENCY = 10
    KEEP_LAST_N_CHECKPOINTS = 3
    LOG_LEVEL = "INFO"
    NUM_RUNS = 3
    PLOT_METRICS = ["accuracy", "loss", "participation_rate", "system_activity", "crc", "ipr"]
    PLOT_FORMATS = ["png", "pdf"]
    
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

# 初始化目录 / Initialize directories
setup_directories()