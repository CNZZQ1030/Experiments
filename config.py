"""
config.py (Updated for CGSV)
配置文件 / Configuration File
包含所有实验参数和系统设置，已更新为CGSV方法
Contains all experimental parameters and system settings, updated for CGSV method
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
    DISTRIBUTION_TYPE = "iid"  # 数据分布类型 / Data distribution type: "iid" or "non-iid"
    NON_IID_ALPHA = 0.5  # Non-IID分布的Dirichlet参数 / Dirichlet parameter for Non-IID
    
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
    # 时间片的核心作用：防止训练后期贡献度递减导致客户端参与度下降
    # Core purpose: Prevent participation decline due to diminishing contribution in late training
    
    TIME_SLICE_TYPE = "rounds"  # 时间片类型 / Time slice type
    # 可选值 / Options: "rounds", "days", "phases", "dynamic", "completion"
    
    ROUNDS_PER_SLICE = 5  # 每个时间片包含的轮次数 / Rounds per time slice
    # 说明：定义一个时间片的长度（以轮次为单位）
    # Description: Defines the length of a time slice (in rounds)
    
    POINTS_VALIDITY_SLICES = 2  # 积分有效期（时间片数） / Points validity period (number of slices)
    # 说明：控制时间窗口大小的关键参数
    # Description: KEY PARAMETER that controls the time window size
    # 例如：POINTS_VALIDITY_SLICES=10, ROUNDS_PER_SLICE=10 
    #      意味着积分有效期为 10×10=100 轮
    # Example: POINTS_VALIDITY_SLICES=10, ROUNDS_PER_SLICE=10
    #         means points are valid for 10×10=100 rounds
    
    DAYS_PER_SLICE = 3  # 基于天数的时间片长度 / Days per time slice (for "days" mode)
    
    # 动态时间片参数 / Dynamic time slice parameters (for "dynamic" mode)
    ACTIVITY_THRESHOLD = 0.5  # 活跃度阈值 / Activity threshold
    BASE_SLICE_LENGTH = 10  # 基础时间片长度 / Base slice length
    
    # ===== 会员等级阈值 / Membership Level Thresholds =====
    # 注意：这些是绝对积分阈值，第3个需求中将改为相对排名
    # Note: These are absolute point thresholds, will be changed to relative ranking in requirement 3
    LEVEL_THRESHOLDS = {
        'bronze': 0,      # 铜级 / Bronze level
        'silver': 2000,   # 银级 / Silver level
        'gold': 6000,     # 金级 / Gold level
        'diamond': 15000  # 钻石级 / Diamond level
    }
    
    # 等级权益倍数 / Level benefit multipliers
    LEVEL_MULTIPLIERS = {
        'bronze': 1.0,
        'silver': 1.2,
        'gold': 1.5,
        'diamond': 2.0
    }
    
    # ===== 差异化模型奖励配置 / Differentiated Model Rewards Configuration =====
    
    # 启用差异化模型奖励 / Enable differentiated model rewards
    ENABLE_TIERED_REWARDS = True
    
    # 等级基础访问比例 / Level base access ratios
    # 与CGSV贡献度结合决定最终访问权限
    # Combined with CGSV contribution to determine final access
    LEVEL_ACCESS_RATIOS = {
        'diamond': 1.0,
        'gold': 0.8,
        'silver': 0.6,
        'bronze': 0.4
    }
    
    # 最小保障访问数量 / Minimum guaranteed access
    MIN_ACCESSIBLE_UPDATES = 1
    
    # 贡献-回报相关性计算参数 / Contribution-Reward Correlation parameters
    CRC_WINDOW_SIZE = 10
    CRC_MIN_SAMPLES = 3
    CRC_START_ROUND = 1

# =====================================
# 数据集配置 / Dataset Configuration
# =====================================

class DatasetConfig:
    """数据集参数配置 / Dataset Parameters"""
    
    # 支持的数据集 / Supported datasets
    AVAILABLE_DATASETS = ["mnist", "fashion-mnist", "cifar10", "cifar100", "shakespeare"]
    
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
    }
    
    NORMALIZE_STD = {
        "mnist": (0.3081,),
        "fashion-mnist": (0.3530,),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
    }
    
    # 输入维度 / Input dimensions
    INPUT_SHAPE = {
        "mnist": (1, 28, 28),
        "fashion-mnist": (1, 28, 28),
        "cifar10": (3, 32, 32),
        "cifar100": (3, 32, 32),
    }
    
    # 类别数 / Number of classes
    NUM_CLASSES = {
        "mnist": 10,
        "fashion-mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "shakespeare": 80,  # 字符级别 / Character level
    }

# =====================================
# 模型配置 / Model Configuration
# =====================================

class ModelConfig:
    """模型参数配置 / Model Parameters"""
    
    # CNN模型配置 / CNN model configuration
    CNN_CHANNELS = [32, 64]  # 卷积通道数 / Convolution channels
    CNN_KERNEL_SIZE = 3  # 卷积核大小 / Kernel size
    CNN_DROPOUT = 0.5  # Dropout率 / Dropout rate
    
    # LSTM模型配置（用于Shakespeare数据集） / LSTM model configuration (for Shakespeare dataset)
    LSTM_HIDDEN_SIZE = 128  # 隐藏层大小 / Hidden layer size
    LSTM_NUM_LAYERS = 2  # LSTM层数 / Number of LSTM layers
    LSTM_DROPOUT = 0.2  # Dropout率 / Dropout rate
    EMBEDDING_DIM = 8  # 嵌入维度 / Embedding dimension

# =====================================
# 实验配置 / Experiment Configuration
# =====================================

class ExperimentConfig:
    """实验参数配置 / Experiment Parameters"""
    
    # 实验名称 / Experiment name
    EXPERIMENT_NAME = f"FL_Incentive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 评估频率 / Evaluation frequency
    EVAL_FREQUENCY = 5  # 每多少轮评估一次 / Evaluate every N rounds
    
    # 保存频率 / Save frequency
    SAVE_FREQUENCY = 10  # 每多少轮保存一次 / Save every N rounds
    
    # 是否只保留最近N个检查点 / Whether to keep only the last N checkpoints
    KEEP_LAST_N_CHECKPOINTS = 3  # 设为None则保留所有 / Set to None to keep all
    
    # 日志级别 / Logging level
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # 实验重复次数 / Number of experiment repetitions
    NUM_RUNS = 3
    
    # 可视化配置 / Visualization configuration
    PLOT_METRICS = ["accuracy", "loss", "participation_rate", "system_activity", "crc", "ipr"]
    PLOT_FORMATS = ["png", "pdf"]  # 图片保存格式 / Image save formats
    
    # 输出路径配置
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