"""
数据集加载和预处理模块 / Dataset Loading and Preprocessing Module
支持MNIST, Fashion-MNIST, CIFAR-10/100, Shakespeare等数据集
每个客户端拥有独立的训练集和测试集
Each client has independent training and test sets
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import random

class FederatedDataLoader:
    """
    联邦学习数据加载器 / Federated Learning Data Loader
    负责数据集的加载、分割和分发 / Responsible for dataset loading, splitting and distribution
    每个客户端拥有独立的训练集和测试集 / Each client has independent train/test sets
    """
    
    def __init__(self, dataset_name: str, num_clients: int, 
                 batch_size: int, data_root: str = "./data",
                 distribution: str = "iid", alpha: float = 0.5,
                 test_ratio: float = 0.2):
        """
        初始化数据加载器 / Initialize data loader
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            num_clients: 客户端数量 / Number of clients
            batch_size: 批次大小 / Batch size
            data_root: 数据根目录 / Data root directory
            distribution: 数据分布类型 ("iid" or "non-iid") / Data distribution type
            alpha: Dirichlet分布参数(用于non-iid) / Dirichlet distribution parameter (for non-iid)
            test_ratio: 测试集比例 / Test set ratio (default 0.2 = 20%)
        """
        self.dataset_name = dataset_name.lower()
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.data_root = data_root
        self.distribution = distribution
        self.alpha = alpha
        self.test_ratio = test_ratio
        
        # 加载数据集 / Load dataset
        self.train_dataset, _ = self._load_dataset()  # 只使用原始训练集
        
        # 获取数据集信息 / Get dataset information
        self.num_total_samples = len(self.train_dataset)
        
        # 为每个客户端创建训练集和测试集索引 / Create train/test indices for each client
        self.client_train_indices = {}
        self.client_test_indices = {}
        self._create_client_data_splits()
        
        print(f"\n{'='*70}")
        print(f"Data Distribution: {distribution.upper()}")
        print(f"Total samples: {self.num_total_samples}")
        print(f"Test ratio per client: {test_ratio*100:.0f}%")
        print(f"{'='*70}")
        
    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        加载指定的数据集 / Load the specified dataset
        
        Returns:
            训练集和测试集（测试集在这里不使用）/ Training set and test set (test set not used here)
        """
        if self.dataset_name == "mnist":
            return self._load_mnist()
        elif self.dataset_name == "fashion-mnist":
            return self._load_fashion_mnist()
        elif self.dataset_name == "cifar10":
            return self._load_cifar10()
        elif self.dataset_name == "cifar100":
            return self._load_cifar100()
        elif self.dataset_name == "shakespeare":
            return self._load_shakespeare()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _load_mnist(self) -> Tuple[Dataset, Dataset]:
        """加载MNIST数据集 / Load MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform
        )
        
        return train_dataset, test_dataset
    
    def _load_fashion_mnist(self) -> Tuple[Dataset, Dataset]:
        """加载Fashion-MNIST数据集 / Load Fashion-MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = datasets.FashionMNIST(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.FashionMNIST(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform
        )
        
        return train_dataset, test_dataset
    
    def _load_cifar10(self) -> Tuple[Dataset, Dataset]:
        """加载CIFAR-10数据集 / Load CIFAR-10 dataset"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform_test
        )
        
        return train_dataset, test_dataset
    
    def _load_cifar100(self) -> Tuple[Dataset, Dataset]:
        """加载CIFAR-100数据集 / Load CIFAR-100 dataset"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        
        test_dataset = datasets.CIFAR100(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform_test
        )
        
        return train_dataset, test_dataset
    
    def _load_shakespeare(self) -> Tuple[Dataset, Dataset]:
        """
        加载Shakespeare数据集 / Load Shakespeare dataset
        这里简化为一个示例实现 / This is a simplified example implementation
        """
        class ShakespeareDataset(Dataset):
            """简化的Shakespeare数据集类 / Simplified Shakespeare dataset class"""
            
            def __init__(self, train=True):
                self.data = torch.randn(1000, 80)
                self.targets = torch.randint(0, 80, (1000,))
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        return ShakespeareDataset(train=True), ShakespeareDataset(train=False)
    
    def _create_client_data_splits(self) -> None:
        """
        为每个客户端创建独立的训练集和测试集 / Create independent train/test sets for each client
        """
        if self.distribution == "iid":
            self._create_iid_splits()
        else:
            self._create_non_iid_splits()
    
    def _create_iid_splits(self) -> None:
        """
        创建IID数据分布的训练/测试集分割 / Create train/test splits for IID data distribution
        """
        all_indices = list(range(self.num_total_samples))
        random.shuffle(all_indices)
        
        # 平均分配数据 / Evenly distribute data
        samples_per_client = self.num_total_samples // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            if i == self.num_clients - 1:
                end_idx = self.num_total_samples
            
            client_indices = all_indices[start_idx:end_idx]
            
            # 划分训练集和测试集 / Split into train and test
            num_test = int(len(client_indices) * self.test_ratio)
            random.shuffle(client_indices)
            
            self.client_test_indices[i] = client_indices[:num_test]
            self.client_train_indices[i] = client_indices[num_test:]
            
            print(f"Client {i}: Train={len(self.client_train_indices[i])}, Test={len(self.client_test_indices[i])}")
    
    def _create_non_iid_splits(self) -> None:
        """
        创建Non-IID数据分布的训练/测试集分割（使用Dirichlet分布）
        Create train/test splits for Non-IID data distribution (using Dirichlet distribution)
        """
        # 获取标签 / Get labels
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)
        else:
            labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        
        num_classes = len(np.unique(labels))
        
        # 使用Dirichlet分布生成客户端数据分配 / Use Dirichlet distribution
        client_all_indices = {i: [] for i in range(self.num_clients)}
        
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_splits = np.split(idx_k, proportions)
            for i, idx_split in enumerate(idx_splits):
                client_all_indices[i].extend(idx_split.tolist())
        
        # 为每个客户端划分训练集和测试集 / Split train/test for each client
        for i in range(self.num_clients):
            client_indices = client_all_indices[i]
            random.shuffle(client_indices)
            
            num_test = int(len(client_indices) * self.test_ratio)
            
            self.client_test_indices[i] = client_indices[:num_test]
            self.client_train_indices[i] = client_indices[num_test:]
            
            print(f"Client {i}: Train={len(self.client_train_indices[i])}, Test={len(self.client_test_indices[i])}")
    
    def get_client_train_dataloader(self, client_id: int) -> DataLoader:
        """
        获取指定客户端的训练数据加载器 / Get training data loader for specified client
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            客户端的训练数据加载器 / Client's training data loader
        """
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
        """
        获取指定客户端的测试数据加载器 / Get test data loader for specified client
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            客户端的测试数据加载器 / Client's test data loader
        """
        if client_id not in self.client_test_indices:
            raise ValueError(f"Invalid client ID: {client_id}")
        
        indices = self.client_test_indices[client_id]
        sampler = SubsetRandomSampler(indices)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size * 2,  # 测试时可以使用更大的批次
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            shuffle=False
        )
    
    def get_client_data_info(self, client_id: int) -> Dict:
        """
        获取客户端数据信息 / Get client data information
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            包含数据量、类别分布等信息的字典 / Dictionary containing data size, class distribution, etc.
        """
        train_indices = self.client_train_indices[client_id]
        test_indices = self.client_test_indices[client_id]
        
        # 获取训练集标签分布 / Get training set label distribution
        if hasattr(self.train_dataset, 'targets'):
            train_labels = np.array(self.train_dataset.targets)[train_indices]
            test_labels = np.array(self.train_dataset.targets)[test_indices]
        else:
            train_labels = np.array([self.train_dataset[i][1] for i in train_indices])
            test_labels = np.array([self.train_dataset[i][1] for i in test_indices])
        
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        
        return {
            'num_train_samples': len(train_indices),
            'num_test_samples': len(test_indices),
            'train_label_distribution': dict(zip(train_unique.tolist(), train_counts.tolist())),
            'test_label_distribution': dict(zip(test_unique.tolist(), test_counts.tolist())),
            'train_indices': train_indices,
            'test_indices': test_indices
        }
    
    def visualize_data_distribution(self) -> None:
        """
        可视化数据分布 / Visualize data distribution
        用于检查IID和Non-IID设置的效果 / Used to check the effect of IID and Non-IID settings
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        # 可视化前10个客户端的数据分布 / Visualize data distribution of first 10 clients
        for i in range(min(10, self.num_clients)):
            info = self.get_client_data_info(i)
            label_dist = info['train_label_distribution']
            
            axes[i].bar(label_dist.keys(), label_dist.values())
            axes[i].set_title(f'Client {i}\nTrain:{info["num_train_samples"]}, Test:{info["num_test_samples"]}')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
        
        plt.suptitle(f'{self.dataset_name} - {self.distribution.upper()} Distribution (Train Set)')
        plt.tight_layout()
        plt.savefig(f'data_distribution_{self.dataset_name}_{self.distribution}.png')
        plt.show()