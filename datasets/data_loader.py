"""
datasets/data_loader.py (Extended Version)
数据集加载和预处理模块 - 扩展版本 / Dataset Loading and Preprocessing Module - Extended Version

支持的数据集 / Supported Datasets:
- MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SST

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


class SSTDataset(Dataset):
    """
    Stanford Sentiment Treebank (SST-2) 数据集
    Stanford Sentiment Treebank (SST-2) Dataset
    
    二分类情感分析任务 / Binary sentiment classification task
    """
    
    def __init__(self, data_root: str = "./data", train: bool = True, 
                 max_seq_length: int = 200, vocab_size: int = 20000):
        """
        初始化SST数据集 / Initialize SST dataset
        
        Args:
            data_root: 数据根目录 / Data root directory
            train: 是否为训练集 / Whether training set
            max_seq_length: 最大序列长度 / Maximum sequence length
            vocab_size: 词汇表大小 / Vocabulary size
        """
        self.data_root = data_root
        self.train = train
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        # 下载并加载数据 / Download and load data
        self.texts, self.labels = self._load_sst_data()
        
        # 构建词汇表 / Build vocabulary
        self.word2idx = self._build_vocab()
        
        # 编码文本 / Encode texts
        self.encoded_texts = self._encode_texts()
        
    def _load_sst_data(self) -> Tuple[List[str], List[int]]:
        """
        加载SST数据 / Load SST data
        使用简化版本，可替换为真实SST数据集
        Using simplified version, can be replaced with real SST dataset
        
        Returns:
            (texts, labels) - 文本列表和标签列表 / Text list and label list
        """
        sst_dir = os.path.join(self.data_root, "sst")
        os.makedirs(sst_dir, exist_ok=True)
        
        # 检查是否存在预处理数据 / Check if preprocessed data exists
        data_file = os.path.join(sst_dir, "train.txt" if self.train else "test.txt")
        
        if os.path.exists(data_file):
            # 加载已有数据 / Load existing data
            texts, labels = [], []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        labels.append(int(parts[0]))
                        texts.append(parts[1])
            return texts, labels
        
        # 生成模拟数据（实际使用时替换为真实数据）
        # Generate simulated data (replace with real data in actual use)
        print(f"Generating simulated SST {'train' if self.train else 'test'} data...")
        
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
        
        num_samples = 8000 if self.train else 2000
        texts, labels = [], []
        
        for _ in range(num_samples):
            # 随机生成句子 / Randomly generate sentences
            label = random.randint(0, 1)
            
            # 根据标签选择词汇 / Select words based on label
            if label == 1:  # 正面 / Positive
                sentiment_words = random.sample(positive_words, random.randint(2, 4))
            else:  # 负面 / Negative
                sentiment_words = random.sample(negative_words, random.randint(2, 4))
            
            filler_words = random.sample(neutral_words, random.randint(3, 6))
            all_words = sentiment_words + filler_words
            random.shuffle(all_words)
            
            text = ' '.join(all_words)
            texts.append(text)
            labels.append(label)
        
        # 保存数据 / Save data
        with open(data_file, 'w', encoding='utf-8') as f:
            for label, text in zip(labels, texts):
                f.write(f"{label}\t{text}\n")
        
        return texts, labels
    
    def _build_vocab(self) -> Dict[str, int]:
        """
        构建词汇表 / Build vocabulary
        
        Returns:
            word2idx - 词到索引的映射 / Word to index mapping
        """
        word_freq = {}
        for text in self.texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序 / Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词汇表 / Build vocabulary
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
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
            words = text.lower().split()
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


class FederatedDataLoader:
    """
    联邦学习数据加载器 - 扩展版本 / Federated Learning Data Loader - Extended Version
    
    支持多种数据集和分布类型
    Supports multiple datasets and distribution types
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
            num_clients: 客户端数量 / Number of clients
            batch_size: 批次大小 / Batch size
            data_root: 数据根目录 / Data root directory
            distribution: 数据分布类型 / Data distribution type
                - "iid": 独立同分布 / IID
                - "non-iid-dir": Dirichlet分布 / Dirichlet
                - "non-iid-size": 数据量不平衡 / Imbalanced size
                - "non-iid-class": 类别数不平衡 / Imbalanced class
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
    
    def _load_sst(self) -> Tuple[Dataset, Dataset]:
        """加载SST数据集 / Load SST dataset"""
        train_dataset = SSTDataset(
            data_root=self.data_root, 
            train=True,
            max_seq_length=200,
            vocab_size=20000
        )
        test_dataset = SSTDataset(
            data_root=self.data_root, 
            train=False,
            max_seq_length=200,
            vocab_size=20000
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
        # 使用指数分布创建不平衡性 / Use exponential distribution
        ratios = np.random.exponential(scale=1.0, size=self.num_clients)
        
        # 缩放比例以满足不平衡比例要求 / Scale ratios to meet imbalance requirement
        ratios = ratios / np.min(ratios)  # 最小值为1
        if np.max(ratios) > self.size_imbalance_ratio:
            ratios = ratios / np.max(ratios) * self.size_imbalance_ratio
        
        # 归一化 / Normalize
        ratios = ratios / np.sum(ratios)
        
        # 分配训练集 / Distribute training set
        train_indices = list(range(self.num_train_samples))
        random.shuffle(train_indices)
        
        train_sizes = (ratios * self.num_train_samples).astype(int)
        # 确保总数正确 / Ensure total is correct
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
        
        # 打印统计信息 / Print statistics
        train_counts = [len(self.client_train_indices[i]) for i in range(self.num_clients)]
        print(f"\n  Train samples - Min: {min(train_counts)}, Max: {max(train_counts)}, "
              f"Ratio: {max(train_counts)/max(1, min(train_counts)):.2f}")
    
    def _create_class_imbalanced_splits(self) -> None:
        """
        创建类别数不平衡分布 / Create imbalanced class number distribution
        不同客户端拥有不同数量的类别 / Different clients have different number of classes
        """
        print(f"\nNon-IID Class Imbalanced Distribution / 类别数不平衡分布:")
        print(f"  Classes per Client: {self.min_classes_per_client}-{self.max_classes_per_client}")
        
        # 获取标签 / Get labels
        train_labels = self._get_labels(self.train_dataset)
        test_labels = self._get_labels(self.test_dataset)
        
        num_classes = len(np.unique(train_labels))
        
        # 确保参数有效 / Ensure valid parameters
        min_classes = min(self.min_classes_per_client, num_classes)
        max_classes = min(self.max_classes_per_client, num_classes)
        
        # 为每个客户端随机分配类别 / Randomly assign classes to each client
        client_train_indices = {i: [] for i in range(self.num_clients)}
        client_test_indices = {i: [] for i in range(self.num_clients)}
        
        # 按类别组织数据 / Organize data by class
        train_class_indices = {k: np.where(train_labels == k)[0].tolist() 
                              for k in range(num_classes)}
        test_class_indices = {k: np.where(test_labels == k)[0].tolist() 
                             for k in range(num_classes)}
        
        # 为每个客户端分配类别 / Assign classes to each client
        for i in range(self.num_clients):
            # 随机决定该客户端的类别数 / Randomly decide number of classes
            num_client_classes = random.randint(min_classes, max_classes)
            
            # 随机选择类别 / Randomly select classes
            selected_classes = random.sample(range(num_classes), num_client_classes)
            
            # 从选定类别中获取数据 / Get data from selected classes
            for class_idx in selected_classes:
                # 计算每个类别分配给该客户端的数据量 / Calculate data per class per client
                train_per_class = len(train_class_indices[class_idx]) // self.num_clients
                test_per_class = len(test_class_indices[class_idx]) // self.num_clients
                
                # 随机选择样本 / Randomly select samples
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
            
            # 打乱数据 / Shuffle data
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
    
    def get_client_data_info(self, client_id: int) -> Dict:
        """获取客户端数据信息 / Get client data information"""
        train_indices = self.client_train_indices[client_id]
        test_indices = self.client_test_indices[client_id]
        
        train_labels = self._get_labels(self.train_dataset)[train_indices] if train_indices else []
        test_labels = self._get_labels(self.test_dataset)[test_indices] if test_indices else []
        
        train_unique, train_counts = np.unique(train_labels, return_counts=True) if len(train_labels) > 0 else ([], [])
        test_unique, test_counts = np.unique(test_labels, return_counts=True) if len(test_labels) > 0 else ([], [])
        
        return {
            'num_train_samples': len(train_indices),
            'num_test_samples': len(test_indices),
            'train_label_distribution': dict(zip(train_unique.tolist(), train_counts.tolist())) if len(train_unique) > 0 else {},
            'test_label_distribution': dict(zip(test_unique.tolist(), test_counts.tolist())) if len(test_unique) > 0 else {},
            'train_indices': train_indices,
            'test_indices': test_indices
        }
    
    def visualize_data_distribution(self, save_path: str = None) -> None:
        """可视化数据分布 / Visualize data distribution"""
        import matplotlib.pyplot as plt
        
        num_plots = min(10, self.num_clients)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(num_plots):
            info = self.get_client_data_info(i)
            label_dist = info['train_label_distribution']
            
            if label_dist:
                axes[i].bar(label_dist.keys(), label_dist.values())
            axes[i].set_title(f'Client {i}\nTrain:{info["num_train_samples"]}, Test:{info["num_test_samples"]}')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
        
        plt.suptitle(f'{self.dataset_name} - {self.distribution.upper()} Distribution')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution visualization saved to: {save_path}")
        else:
            plt.savefig(f'data_distribution_{self.dataset_name}_{self.distribution}.png')
        
        plt.close()