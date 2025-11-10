"""
配置文件 / Configuration File (已修正差异化模型配置)
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
    """联邦学习参数配置 / Federated Learning Parameters"""
    
    # 客户端配置 / Client configuration
    NUM_CLIENTS = 100  # 客户端总数 / Total number of clients
    CLIENTS_PER_ROUND = 30  # 每轮选择的客户端数 / Number of clients selected per round
    
    # 训练配置 / Training configuration
    NUM_ROUNDS = 100  # 训练轮次 / Number of training rounds
    LOCAL_EPOCHS = 5  # 本地训练轮次 / Local training epochs
    LOCAL_BATCH_SIZE = 32  # 本地批次大小 / Local batch size
    LEARNING_RATE = 0.01  # 学习率 / Learning rate
    
    # 数据分布 / Data distribution
    DATA_DISTRIBUTION = "iid"  # "iid" or "non-iid"
    NON_IID_ALPHA = 0.5  # Dirichlet分布参数 / Dirichlet distribution parameter
    
    # 客户端选择策略 / Client selection strategy
    USE_DYNAMIC_SELECTION = True  # 是否使用动态选择（基于等级和积分）/ Whether to use dynamic selection based on level and points

# =====================================
# 激励机制配置 / Incentive Mechanism Configuration
# =====================================

class IncentiveConfig:
    """激励机制参数配置 / Incentive Mechanism Parameters"""
    
    # 积分计算权重 / Points calculation weights
    ALPHA = 0.3  # 数据量权重 / Data size weight
    BETA = 0.3   # 计算时间权重 / Computation time weight
    GAMMA = 0.4  # 模型质量权重 / Model quality weight
    
    # 会员等级阈值 / Membership level thresholds
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
    
    # 时间片配置 / Time slice configuration
    TIME_SLICE_TYPE = "rounds"  # "rounds", "days", "phases", "dynamic", "completion"
    ROUNDS_PER_SLICE = 10  # 基于轮次的时间片长度 / Rounds per time slice
    DAYS_PER_SLICE = 3  # 基于天数的时间片长度 / Days per time slice
    
    # 积分有效期（时间片数） / Points validity period (number of time slices)
    POINTS_VALIDITY_SLICES = 10
    
    # 动态时间片参数 / Dynamic time slice parameters
    ACTIVITY_THRESHOLD = 0.5  # 活跃度阈值 / Activity threshold
    BASE_SLICE_LENGTH = 10  # 基础时间片长度 / Base slice length
    
    # ===== 差异化模型奖励配置 / Differentiated Model Rewards Configuration =====
    
    # 启用差异化模型奖励 / Enable differentiated model rewards
    ENABLE_TIERED_REWARDS = True
    
    # ===== 差异化策略说明 / Differentiation Strategy =====
    # 
    # 实现方式: 基于访问控制的个性化模型聚合
    # Implementation: Personalized model aggregation based on access control
    # 
    # 核心机制 / Core Mechanism:
    # 1. 贡献度 = 0: 
    #    - 只能使用自己的本地模型（最低保障）
    #    - Only get local model (minimum guarantee)
    # 
    # 2. 贡献度 > 0:
    #    - 可以访问其他客户端的模型更新
    #    - Can access other clients' model updates
    #    - 访问数量由贡献度和会员等级共同决定
    #    - Number of accessible updates determined by contribution and level
    # 
    # 3. 聚合方式:
    #    - 对可访问的更新进行加权聚合（按贡献度加权）
    #    - Weighted aggregation of accessible updates (weighted by contribution)
    #    - 贡献度高的更新获得更大权重
    #    - Updates with higher contribution get larger weights
    #
    # 示例 / Example:
    # - Bronze级，贡献度0.2: 只能访问最好的1-2个更新
    #   Bronze level, contribution 0.2: Access only 1-2 best updates
    # - Diamond级，贡献度0.8: 可以访问几乎所有更新
    #   Diamond level, contribution 0.8: Access almost all updates
    # 
    # ===================================================================
    
    # 等级访问比例 / Level access ratios
    # 控制不同等级客户端能访问的更新比例
    # Control the ratio of updates accessible by different levels
    LEVEL_ACCESS_RATIOS = {
        'diamond': 1.0,   # 可访问100%的可用更新 / Access 100% of available updates
        'gold': 0.8,      # 可访问80%的可用更新 / Access 80% of available updates
        'silver': 0.6,    # 可访问60%的可用更新 / Access 60% of available updates
        'bronze': 0.4     # 可访问40%的可用更新 / Access 40% of available updates
    }
    
    # 有效访问比例计算权重 / Effective access ratio calculation weights
    # 有效访问比例 = 贡献度 × CONTRIBUTION_WEIGHT + 等级比例 × LEVEL_WEIGHT
    # Effective access ratio = contribution × CONTRIBUTION_WEIGHT + level_ratio × LEVEL_WEIGHT
    CONTRIBUTION_WEIGHT = 0.7  # 贡献度权重70% / Contribution weight 70%
    LEVEL_WEIGHT = 0.3         # 等级权重30% / Level weight 30%
    
    # 最小保障访问数量 / Minimum guaranteed access
    # 即使贡献度很低，也至少能访问这么多更新（包括自己的）
    # Even with low contribution, can access at least this many updates (including own)
    MIN_ACCESSIBLE_UPDATES = 1
    
    # 贡献-回报相关性计算参数
    # Contribution-Reward Correlation parameters
    CRC_WINDOW_SIZE = 10  # 计算相关性的窗口大小（轮次数）
    CRC_MIN_SAMPLES = 3   # 计算相关性所需的最小样本数
    CRC_START_ROUND = 1   # 从第1轮开始尝试计算CRC

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
    PLOT_METRICS = ["accuracy", "loss", "participation_rate", "system_activity", "crc"]
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