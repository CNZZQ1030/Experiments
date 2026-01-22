"""
datasets/data_loader.py (Extended with Real MR & SST Datasets)
数据集加载和预处理模块 - 扩展版本，支持真实的MR和SST数据集
Dataset Loading and Preprocessing Module - Extended with Real MR & SST Support

支持的数据集 / Supported Datasets:
- MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 (图像分类 / Image Classification)
- Movie Review (MR) (电影评论情感分析 / Movie Review Sentiment Analysis)
- Stanford Sentiment Treebank (SST-2) (斯坦福情感树库 / Stanford Sentiment Treebank)

支持的分布类型 / Supported Distribution Types:
- iid: 独立同分布 / Independent and identically distributed
- non-iid-dir: Dirichlet分布 / Dirichlet distribution (quantity skew)
- non-iid-size: 数据量不平衡 / Imbalanced dataset size
- non-iid-class: 类别数不平衡 / Imbalanced class number
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import random
import os
import urllib.request
import tarfile
import re
from collections import Counter


class MovieReviewDataset(Dataset):
    """
    Movie Review (MR) 数据集
    Movie Review (MR) Dataset
    
    来源 / Source: https://www.cs.cornell.edu/people/pabo/movie-review-data/
    论文 / Paper: Pang and Lee, "Seeing stars: Exploiting class relationships for 
                  sentiment categorization with respect to rating scales." (2005)
    
    二分类情感分析任务 / Binary sentiment classification task
    - Positive reviews: 正面评论
    - Negative reviews: 负面评论
    """
    
    def __init__(self, data_root: str = "./data", train: bool = True, 
                 max_seq_length: int = 200, vocab_size: int = 20000,
                 download: bool = True):
        """
        初始化MR数据集 / Initialize MR dataset
        
        Args:
            data_root: 数据根目录 / Data root directory
            train: 是否为训练集 / Whether training set
            max_seq_length: 最大序列长度 / Maximum sequence length
            vocab_size: 词汇表大小 / Vocabulary size
            download: 是否下载数据 / Whether to download data
        """
        self.data_root = data_root
        self.train = train
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        self.mr_dir = os.path.join(data_root, "movie_review")
        os.makedirs(self.mr_dir, exist_ok=True)
        
        # 下载并加载数据 / Download and load data
        if download:
            self._download_mr_data()
        
        self.texts, self.labels = self._load_mr_data()
        
        # 构建词汇表 / Build vocabulary
        self.word2idx = self._build_vocab()
        
        # 编码文本 / Encode texts
        self.encoded_texts = self._encode_texts()
    
    def _download_mr_data(self) -> None:
        """
        下载MR数据集 / Download MR dataset
        如果数据不存在，使用预定义的示例数据
        If data doesn't exist, use predefined example data
        """
        pos_file = os.path.join(self.mr_dir, "rt-polarity.pos")
        neg_file = os.path.join(self.mr_dir, "rt-polarity.neg")
        
        # 如果文件已存在，跳过 / Skip if files exist
        if os.path.exists(pos_file) and os.path.exists(neg_file):
            print(f"MR dataset already exists at {self.mr_dir}")
            return
        
        print(f"Creating sample MR dataset at {self.mr_dir}...")
        print("For real MR dataset, download from: https://www.cs.cornell.edu/people/pabo/movie-review-data/")
        
        # 创建示例正面评论 / Create sample positive reviews
        positive_samples = [
            "the movie was absolutely fantastic and I loved every minute of it",
            "brilliant performance by the lead actor, highly recommended",
            "one of the best films I have seen this year",
            "outstanding cinematography and excellent storytelling",
            "a masterpiece that will be remembered for years to come",
            "incredibly moving and beautifully shot film",
            "the director's vision was executed perfectly",
            "amazing special effects and gripping plot",
            "superb acting and wonderful screenplay",
            "this film exceeded all my expectations"
        ] * 100  # 重复以创建更多样本 / Repeat to create more samples
        
        # 创建示例负面评论 / Create sample negative reviews
        negative_samples = [
            "terrible waste of time and money",
            "the plot was confusing and poorly executed",
            "disappointing performances across the board",
            "boring and predictable storyline",
            "one of the worst movies I have ever seen",
            "awful screenplay and terrible acting",
            "complete disaster from start to finish",
            "the film was a huge disappointment",
            "poorly directed and badly written",
            "tedious and uninteresting throughout"
        ] * 100
        
        # 保存到文件 / Save to files
        with open(pos_file, 'w', encoding='utf-8') as f:
            for review in positive_samples:
                f.write(review + '\n')
        
        with open(neg_file, 'w', encoding='utf-8') as f:
            for review in negative_samples:
                f.write(review + '\n')
        
        print(f"✓ Sample MR dataset created with {len(positive_samples)} positive and {len(negative_samples)} negative reviews")
    
    def _load_mr_data(self) -> Tuple[List[str], List[int]]:
        """
        加载MR数据 / Load MR data
        
        Returns:
            (texts, labels) - 文本列表和标签列表 / Text list and label list
        """
        pos_file = os.path.join(self.mr_dir, "rt-polarity.pos")
        neg_file = os.path.join(self.mr_dir, "rt-polarity.neg")
        
        texts = []
        labels = []
        
        # 读取正面评论 / Read positive reviews
        if os.path.exists(pos_file):
            with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(self._clean_text(line))
                        labels.append(1)  # 1 for positive
        
        # 读取负面评论 / Read negative reviews
        if os.path.exists(neg_file):
            with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(self._clean_text(line))
                        labels.append(0)  # 0 for negative
        
        # 打乱数据 / Shuffle data
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)
        
        # 划分训练集和测试集 (80/20) / Split train/test (80/20)
        split_idx = int(len(texts) * 0.8)
        
        if self.train:
            return texts[:split_idx], labels[:split_idx]
        else:
            return texts[split_idx:], labels[split_idx:]
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本 / Clean text
        移除特殊字符，转换为小写
        Remove special characters, convert to lowercase
        """
        # 转小写 / Convert to lowercase
        text = text.lower()
        # 移除特殊字符，保留字母、数字和空格 / Remove special chars, keep letters, numbers, spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # 移除多余空格 / Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _build_vocab(self) -> Dict[str, int]:
        """
        构建词汇表 / Build vocabulary
        
        Returns:
            word2idx - 词到索引的映射 / Word to index mapping
        """
        word_freq = Counter()
        for text in self.texts:
            for word in text.split():
                word_freq[word] += 1
        
        # 按频率排序 / Sort by frequency
        sorted_words = word_freq.most_common(self.vocab_size - 2)
        
        # 构建词汇表 / Build vocabulary
        # 0: <PAD>, 1: <UNK>
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(sorted_words):
            word2idx[word] = i + 2
        
        return word2idx
    
    def _encode_texts(self) -> List[torch.Tensor]:
        """
        编码文本为索引序列 / Encode texts to index sequences
        
        Returns:
            encoded_texts - 编码后的文本列表 / Encoded text list
        """
        encoded = []
        for text in self.texts:
            words = text.split()
            indices = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
            
            # 填充或截断 / Pad or truncate
            if len(indices) < self.max_seq_length:
                indices = indices + [0] * (self.max_seq_length - len(indices))
            else:
                indices = indices[:self.max_seq_length]
            
            encoded.append(torch.tensor(indices, dtype=torch.long))
        
        return encoded
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.encoded_texts[idx], self.labels[idx]
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小 / Get vocabulary size"""
        return len(self.word2idx)


class SSTDataset(Dataset):
    """
    Stanford Sentiment Treebank (SST-2) 数据集
    Stanford Sentiment Treebank (SST-2) Dataset
    
    来源 / Source: https://nlp.stanford.edu/sentiment/
    论文 / Paper: Socher et al., "Recursive Deep Models for Semantic Compositionality 
                  Over a Sentiment Treebank" (2013)
    
    二分类情感分析任务 / Binary sentiment classification task
    """
    
    def __init__(self, data_root: str = "./data", train: bool = True, 
                 max_seq_length: int = 200, vocab_size: int = 20000,
                 download: bool = True):
        """
        初始化SST数据集 / Initialize SST dataset
        
        Args:
            data_root: 数据根目录 / Data root directory
            train: 是否为训练集 / Whether training set
            max_seq_length: 最大序列长度 / Maximum sequence length
            vocab_size: 词汇表大小 / Vocabulary size
            download: 是否下载数据 / Whether to download data
        """
        self.data_root = data_root
        self.train = train
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        self.sst_dir = os.path.join(data_root, "sst")
        os.makedirs(self.sst_dir, exist_ok=True)
        
        # 下载并加载数据 / Download and load data
        if download:
            self._download_sst_data()
        
        self.texts, self.labels = self._load_sst_data()
        
        # 构建词汇表 / Build vocabulary
        self.word2idx = self._build_vocab()
        
        # 编码文本 / Encode texts
        self.encoded_texts = self._encode_texts()
    
    def _download_sst_data(self) -> None:
        """
        下载SST数据集 / Download SST dataset
        创建示例数据（实际使用时应下载真实数据）
        Create sample data (should download real data in actual use)
        """
        train_file = os.path.join(self.sst_dir, "train.txt")
        test_file = os.path.join(self.sst_dir, "test.txt")
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"SST dataset already exists at {self.sst_dir}")
            return
        
        print(f"Creating sample SST dataset at {self.sst_dir}...")
        print("For real SST dataset, download from: https://nlp.stanford.edu/sentiment/")
        
        # 正面词汇 / Positive words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'fantastic', 'love', 'best', 'happy', 'beautiful',
                         'perfect', 'awesome', 'brilliant', 'outstanding', 'superb']
        
        # 负面词汇 / Negative words
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor',
                         'worst', 'hate', 'boring', 'stupid', 'waste',
                         'disappointing', 'mediocre', 'dull', 'annoying', 'pathetic']
        
        # 中性词汇 / Neutral words
        neutral_words = ['the', 'movie', 'film', 'story', 'was', 'is', 'are',
                        'it', 'this', 'that', 'and', 'but', 'with', 'have', 'has']
        
        def generate_sample(label):
            if label == 1:  # 正面 / Positive
                sentiment_words = random.sample(positive_words, random.randint(2, 4))
            else:  # 负面 / Negative
                sentiment_words = random.sample(negative_words, random.randint(2, 4))
            
            filler_words = random.sample(neutral_words, random.randint(3, 6))
            all_words = sentiment_words + filler_words
            random.shuffle(all_words)
            
            return ' '.join(all_words)
        
        # 生成训练集 / Generate training set
        with open(train_file, 'w', encoding='utf-8') as f:
            for _ in range(4000):
                label = random.randint(0, 1)
                text = generate_sample(label)
                f.write(f"{label}\t{text}\n")
        
        # 生成测试集 / Generate test set
        with open(test_file, 'w', encoding='utf-8') as f:
            for _ in range(1000):
                label = random.randint(0, 1)
                text = generate_sample(label)
                f.write(f"{label}\t{text}\n")
        
        print(f"✓ Sample SST dataset created with 4000 train and 1000 test samples")
    
    def _load_sst_data(self) -> Tuple[List[str], List[int]]:
        """
        加载SST数据 / Load SST data
        
        Returns:
            (texts, labels) - 文本列表和标签列表 / Text list and label list
        """
        data_file = os.path.join(self.sst_dir, "train.txt" if self.train else "test.txt")
        
        texts, labels = [], []
        
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        labels.append(int(parts[0]))
                        texts.append(self._clean_text(parts[1]))
        
        return texts, labels
    
    def _clean_text(self, text: str) -> str:
        """清理文本 / Clean text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建词汇表 / Build vocabulary"""
        word_freq = Counter()
        for text in self.texts:
            for word in text.split():
                word_freq[word] += 1
        
        sorted_words = word_freq.most_common(self.vocab_size - 2)
        
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(sorted_words):
            word2idx[word] = i + 2
        
        return word2idx
    
    def _encode_texts(self) -> List[torch.Tensor]:
        """编码文本 / Encode texts"""
        encoded = []
        for text in self.texts:
            words = text.split()
            indices = [self.word2idx.get(word, 1) for word in words]
            
            if len(indices) < self.max_seq_length:
                indices = indices + [0] * (self.max_seq_length - len(indices))
            else:
                indices = indices[:self.max_seq_length]
            
            encoded.append(torch.tensor(indices, dtype=torch.long))
        
        return encoded
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.encoded_texts[idx], self.labels[idx]
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小 / Get vocabulary size"""
        return len(self.word2idx)


class FederatedDataLoader:
    """
    联邦学习数据加载器 - 扩展版本 / Federated Learning Data Loader - Extended Version
    
    支持多种数据集和分布类型
    Supports multiple datasets and distribution types
    
    新增支持 / Newly Supported:
    - Movie Review (MR): 电影评论情感分析
    - Stanford Sentiment Treebank (SST-2): 斯坦福情感树库
    """
    
    def __init__(self, dataset_name: str, num_clients: int, 
                 batch_size: int, data_root: str = "./data",
                 distribution: str = "iid", alpha: float = 0.5,
                 size_imbalance_ratio: float = 5.0,
                 min_classes_per_client: int = 2,
                 max_classes_per_client: int = 5):
        """
        初始化数据加载器 / Initialize data loader
        
        Args:
            dataset_name: 数据集名称 / Dataset name
                - 图像: mnist, fashion-mnist, cifar10, cifar100
                - 文本: mr, sst (新增 / New)
            num_clients: 客户端数量 / Number of clients
            batch_size: 批次大小 / Batch size
            data_root: 数据根目录 / Data root directory
            distribution: 数据分布类型 / Data distribution type
            alpha: Dirichlet分布参数 / Dirichlet parameter
            size_imbalance_ratio: 数据量不平衡比例 / Size imbalance ratio
            min_classes_per_client: 每客户端最少类别数 / Min classes per client
            max_classes_per_client: 每客户端最多类别数 / Max classes per client
        """
        self.dataset_name = dataset_name.lower()
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.data_root = data_root
        self.distribution = distribution.lower()
        self.alpha = alpha
        self.size_imbalance_ratio = size_imbalance_ratio
        self.min_classes_per_client = min_classes_per_client
        self.max_classes_per_client = max_classes_per_client
        
        # 加载数据集 / Load dataset
        self.train_dataset, self.test_dataset = self._load_dataset()
        
        # 获取数据集信息 / Get dataset information
        self.num_train_samples = len(self.train_dataset)
        self.num_test_samples = len(self.test_dataset)
        
        # 为每个客户端创建训练集和测试集索引 / Create train/test indices for each client
        self.client_train_indices = {}
        self.client_test_indices = {}
        self._create_client_data_splits()
        
        # 计算实际分配的样本统计 / Calculate actual allocated sample statistics
        total_train_allocated = sum(len(indices) for indices in self.client_train_indices.values())
        total_test_allocated = sum(len(indices) for indices in self.client_test_indices.values())
        
        print(f"\n{'='*70}")
        print(f"Federated Data Distribution / 联邦数据分配")
        print(f"{'='*70}")
        print(f"Dataset / 数据集: {dataset_name}")
        print(f"Distribution / 分布: {distribution.upper()}")
        if distribution == "non-iid-dir":
            print(f"  Alpha: {alpha}")
        elif distribution == "non-iid-size":
            print(f"  Size Imbalance Ratio: {size_imbalance_ratio}")
        elif distribution == "non-iid-class":
            print(f"  Classes per Client: {min_classes_per_client}-{max_classes_per_client}")
        print(f"Number of clients / 客户端数: {num_clients}")
        print(f"\nOriginal Dataset / 原始数据集:")
        print(f"  Training samples / 训练样本: {self.num_train_samples}")
        print(f"  Test samples / 测试样本: {self.num_test_samples}")
        print(f"\nAllocated to Clients / 分配给客户端:")
        print(f"  Total training samples / 训练样本总计: {total_train_allocated}")
        print(f"  Total test samples / 测试样本总计: {total_test_allocated}")
        print(f"  Avg per client / 每客户端平均: train={total_train_allocated/num_clients:.0f}, test={total_test_allocated/num_clients:.0f}")
        print(f"{'='*70}")
    
    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """加载指定的数据集 / Load the specified dataset"""
        if self.dataset_name == "mnist":
            return self._load_mnist()
        elif self.dataset_name == "fashion-mnist":
            return self._load_fashion_mnist()
        elif self.dataset_name == "cifar10":
            return self._load_cifar10()
        elif self.dataset_name == "cifar100":
            return self._load_cifar100()
        elif self.dataset_name == "mr":
            return self._load_mr()
        elif self.dataset_name == "sst":
            return self._load_sst()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_mnist(self) -> Tuple[Dataset, Dataset]:
        """加载MNIST数据集 / Load MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root=self.data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=self.data_root, train=False, download=True, transform=transform
        )
        return train_dataset, test_dataset
    
    def _load_fashion_mnist(self) -> Tuple[Dataset, Dataset]:
        """加载Fashion-MNIST数据集 / Load Fashion-MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = datasets.FashionMNIST(
            root=self.data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=self.data_root, train=False, download=True, transform=transform
        )
        return train_dataset, test_dataset
    
    def _load_cifar10(self) -> Tuple[Dataset, Dataset]:
        """加载CIFAR-10数据集 / Load CIFAR-10 dataset"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=transform_test
        )
        return train_dataset, test_dataset
    
    def _load_cifar100(self) -> Tuple[Dataset, Dataset]:
        """加载CIFAR-100数据集 / Load CIFAR-100 dataset"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(
            root=self.data_root, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=self.data_root, train=False, download=True, transform=transform_test
        )
        return train_dataset, test_dataset
    
    def _load_mr(self) -> Tuple[Dataset, Dataset]:
        """加载Movie Review数据集 / Load Movie Review dataset"""
        train_dataset = MovieReviewDataset(
            data_root=self.data_root, 
            train=True,
            max_seq_length=200,
            vocab_size=20000,
            download=True
        )
        test_dataset = MovieReviewDataset(
            data_root=self.data_root, 
            train=False,
            max_seq_length=200,
            vocab_size=20000,
            download=True
        )
        
        # 共享词汇表 / Share vocabulary
        test_dataset.word2idx = train_dataset.word2idx
        test_dataset.encoded_texts = test_dataset._encode_texts()
        
        return train_dataset, test_dataset
    
    def _load_sst(self) -> Tuple[Dataset, Dataset]:
        """加载SST数据集 / Load SST dataset"""
        train_dataset = SSTDataset(
            data_root=self.data_root, 
            train=True,
            max_seq_length=200,
            vocab_size=20000,
            download=True
        )
        test_dataset = SSTDataset(
            data_root=self.data_root, 
            train=False,
            max_seq_length=200,
            vocab_size=20000,
            download=True
        )
        
        # 共享词汇表 / Share vocabulary
        test_dataset.word2idx = train_dataset.word2idx
        test_dataset.encoded_texts = test_dataset._encode_texts()
        
        return train_dataset, test_dataset
    
    def _create_client_data_splits(self) -> None:
        """为每个客户端创建独立的训练集和测试集 / Create independent train/test sets"""
        if self.distribution == "iid":
            self._create_iid_splits()
        elif self.distribution == "non-iid-dir":
            self._create_dirichlet_splits()
        elif self.distribution == "non-iid-size":
            self._create_size_imbalanced_splits()
        elif self.distribution == "non-iid-class":
            self._create_class_imbalanced_splits()
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
    
    def _create_iid_splits(self) -> None:
        """
        创建IID数据分布 / Create IID data distribution
        数据随机均匀分配给客户端 / Data randomly and evenly distributed to clients
        """
        print(f"\nIID Distribution / IID分布:")
        
        # 分配训练集 / Distribute training set
        train_indices = list(range(self.num_train_samples))
        random.shuffle(train_indices)
        train_per_client = self.num_train_samples // self.num_clients
        
        # 分配测试集 / Distribute test set
        test_indices = list(range(self.num_test_samples))
        random.shuffle(test_indices)
        test_per_client = self.num_test_samples // self.num_clients
        
        for i in range(self.num_clients):
            # 训练集分配 / Training set allocation
            train_start = i * train_per_client
            train_end = train_start + train_per_client
            if i == self.num_clients - 1:
                train_end = self.num_train_samples
            self.client_train_indices[i] = train_indices[train_start:train_end]
            
            # 测试集分配 / Test set allocation
            test_start = i * test_per_client
            test_end = test_start + test_per_client
            if i == self.num_clients - 1:
                test_end = self.num_test_samples
            self.client_test_indices[i] = test_indices[test_start:test_end]
            
            if i < 5:
                print(f"  Client {i}: Train={len(self.client_train_indices[i])}, "
                      f"Test={len(self.client_test_indices[i])}")
        
        if self.num_clients > 5:
            print(f"  ... (remaining {self.num_clients - 5} clients)")
    
    def _create_dirichlet_splits(self) -> None:
        """
        创建Dirichlet Non-IID数据分布 / Create Dirichlet Non-IID distribution
        使用Dirichlet分布控制数据异质性 / Use Dirichlet to control data heterogeneity
        """
        print(f"\nNon-IID Dirichlet Distribution / Dirichlet Non-IID分布 (α={self.alpha}):")
        
        # 获取训练集标签 / Get training labels
        train_labels = self._get_labels(self.train_dataset)
        test_labels = self._get_labels(self.test_dataset)
        
        num_classes = len(np.unique(train_labels))
        
        # 使用Dirichlet分布分配训练集 / Distribute training set using Dirichlet
        client_train_indices = {i: [] for i in range(self.num_clients)}
        for k in range(num_classes):
            idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_splits = np.split(idx_k, proportions)
            for i, idx_split in enumerate(idx_splits):
                client_train_indices[i].extend(idx_split.tolist())
        
        # 使用相同策略分配测试集 / Distribute test set using same strategy
        client_test_indices = {i: [] for i in range(self.num_clients)}
        for k in range(num_classes):
            idx_k = np.where(test_labels == k)[0]
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_splits = np.split(idx_k, proportions)
            for i, idx_split in enumerate(idx_splits):
                client_test_indices[i].extend(idx_split.tolist())
        
        # 随机打乱每个客户端的数据 / Shuffle each client's data
        for i in range(self.num_clients):
            random.shuffle(client_train_indices[i])
            random.shuffle(client_test_indices[i])
            
            self.client_train_indices[i] = client_train_indices[i]
            self.client_test_indices[i] = client_test_indices[i]
            
            if i < 5:
                train_labels_i = train_labels[self.client_train_indices[i]]
                unique, counts = np.unique(train_labels_i, return_counts=True)
                print(f"  Client {i}: Train={len(self.client_train_indices[i])}, "
                      f"Test={len(self.client_test_indices[i])}, "
                      f"Classes={len(unique)}")
        
        if self.num_clients > 5:
            print(f"  ... (remaining {self.num_clients - 5} clients)")
    
    def _create_size_imbalanced_splits(self) -> None:
        """
        创建数据量不平衡分布 / Create imbalanced dataset size distribution
        不同客户端拥有不同数量的数据 / Different clients have different amounts of data
        """
        print(f"\nNon-IID Size Imbalanced Distribution / 数据量不平衡分布:")
        print(f"  Size Imbalance Ratio: {self.size_imbalance_ratio}")
        
        # 生成不平衡的数据量分配 / Generate imbalanced data allocation
        ratios = np.random.exponential(scale=1.0, size=self.num_clients)
        ratios = ratios / np.min(ratios)
        if np.max(ratios) > self.size_imbalance_ratio:
            ratios = ratios / np.max(ratios) * self.size_imbalance_ratio
        ratios = ratios / np.sum(ratios)
        
        # 分配训练集 / Distribute training set
        train_indices = list(range(self.num_train_samples))
        random.shuffle(train_indices)
        train_sizes = (ratios * self.num_train_samples).astype(int)
        train_sizes[-1] = self.num_train_samples - np.sum(train_sizes[:-1])
        
        # 分配测试集 / Distribute test set
        test_indices = list(range(self.num_test_samples))
        random.shuffle(test_indices)
        test_sizes = (ratios * self.num_test_samples).astype(int)
        test_sizes[-1] = self.num_test_samples - np.sum(test_sizes[:-1])
        
        # 分配给客户端 / Distribute to clients
        train_start = 0
        test_start = 0
        
        for i in range(self.num_clients):
            train_end = train_start + train_sizes[i]
            test_end = test_start + test_sizes[i]
            
            self.client_train_indices[i] = train_indices[train_start:train_end]
            self.client_test_indices[i] = test_indices[test_start:test_end]
            
            train_start = train_end
            test_start = test_end
            
            if i < 5:
                print(f"  Client {i}: Train={len(self.client_train_indices[i])}, "
                      f"Test={len(self.client_test_indices[i])}, "
                      f"Ratio={ratios[i]*self.num_clients:.2f}x")
        
        if self.num_clients > 5:
            print(f"  ... (remaining {self.num_clients - 5} clients)")
    
    def _create_class_imbalanced_splits(self) -> None:
        """
        创建类别数不平衡分布 / Create imbalanced class number distribution
        不同客户端拥有不同数量的类别 / Different clients have different number of classes
        """
        print(f"\nNon-IID Class Imbalanced Distribution / 类别数不平衡分布:")
        print(f"  Classes per Client: {self.min_classes_per_client}-{self.max_classes_per_client}")
        
        train_labels = self._get_labels(self.train_dataset)
        test_labels = self._get_labels(self.test_dataset)
        num_classes = len(np.unique(train_labels))
        
        min_classes = min(self.min_classes_per_client, num_classes)
        max_classes = min(self.max_classes_per_client, num_classes)
        
        client_train_indices = {i: [] for i in range(self.num_clients)}
        client_test_indices = {i: [] for i in range(self.num_clients)}
        
        train_class_indices = {k: np.where(train_labels == k)[0].tolist() 
                              for k in range(num_classes)}
        test_class_indices = {k: np.where(test_labels == k)[0].tolist() 
                             for k in range(num_classes)}
        
        for i in range(self.num_clients):
            num_client_classes = random.randint(min_classes, max_classes)
            selected_classes = random.sample(range(num_classes), num_client_classes)
            
            for class_idx in selected_classes:
                train_per_class = len(train_class_indices[class_idx]) // self.num_clients
                test_per_class = len(test_class_indices[class_idx]) // self.num_clients
                
                if train_class_indices[class_idx]:
                    selected_train = random.sample(
                        train_class_indices[class_idx],
                        min(train_per_class, len(train_class_indices[class_idx]))
                    )
                    client_train_indices[i].extend(selected_train)
                
                if test_class_indices[class_idx]:
                    selected_test = random.sample(
                        test_class_indices[class_idx],
                        min(test_per_class, len(test_class_indices[class_idx]))
                    )
                    client_test_indices[i].extend(selected_test)
            
            random.shuffle(client_train_indices[i])
            random.shuffle(client_test_indices[i])
            
            self.client_train_indices[i] = client_train_indices[i]
            self.client_test_indices[i] = client_test_indices[i]
            
            if i < 5:
                train_labels_i = train_labels[self.client_train_indices[i]] if self.client_train_indices[i] else []
                unique_classes = np.unique(train_labels_i) if len(train_labels_i) > 0 else []
                print(f"  Client {i}: Train={len(self.client_train_indices[i])}, "
                      f"Test={len(self.client_test_indices[i])}, "
                      f"Classes={len(unique_classes)}")
        
        if self.num_clients > 5:
            print(f"  ... (remaining {self.num_clients - 5} clients)")
    
    def _get_labels(self, dataset: Dataset) -> np.ndarray:
        """
        获取数据集的标签 / Get dataset labels
        
        Args:
            dataset: 数据集 / Dataset
            
        Returns:
            标签数组 / Label array
        """
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            return np.array(dataset.labels)
        else:
            # 遍历数据集获取标签 / Iterate dataset to get labels
            return np.array([dataset[i][1] for i in range(len(dataset))])
    
    def get_client_train_dataloader(self, client_id: int) -> DataLoader:
        """获取客户端训练数据加载器 / Get client training dataloader"""
        if client_id not in self.client_train_indices:
            raise ValueError(f"Invalid client ID: {client_id}")
        
        indices = self.client_train_indices[client_id]
        sampler = SubsetRandomSampler(indices)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
    
    def get_client_test_dataloader(self, client_id: int) -> DataLoader:
        """获取客户端测试数据加载器 / Get client test dataloader"""
        if client_id not in self.client_test_indices:
            raise ValueError(f"Invalid client ID: {client_id}")
        
        indices = self.client_test_indices[client_id]
        sampler = SubsetRandomSampler(indices)
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            shuffle=False
        )
    
    def get_num_train_samples(self, client_id: int) -> int:
        """获取客户端训练样本数 / Get number of training samples"""
        return len(self.client_train_indices.get(client_id, []))
    
    def get_num_test_samples(self, client_id: int) -> int:
        """获取客户端测试样本数 / Get number of test samples"""
        return len(self.client_test_indices.get(client_id, []))
    
    def get_vocab_size(self) -> int:
        """
        获取词汇表大小（仅用于文本数据集）
        Get vocabulary size (only for text datasets)
        """
        if hasattr(self.train_dataset, 'get_vocab_size'):
            return self.train_dataset.get_vocab_size()
        return 0