"""
数据集加载和预处理模块 / Dataset Loading and Preprocessing Module
支持MNIST, Fashion-MNIST, CIFAR-10/100等数据集
将训练集和测试集都分配给客户端
Both training and test sets are distributed to clients
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
    每个客户端拥有独立的训练集和测试集
    Each client has independent training and test sets
    """
    
    def __init__(self, dataset_name: str, num_clients: int, 
                 batch_size: int, data_root: str = "./data",
                 distribution: str = "iid", alpha: float = 0.5):
        """
        初始化数据加载器 / Initialize data loader
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            num_clients: 客户端数量 / Number of clients
            batch_size: 批次大小 / Batch size
            data_root: 数据根目录 / Data root directory
            distribution: 数据分布类型 ("iid" or "non-iid") / Data distribution type
            alpha: Dirichlet分布参数(用于non-iid) / Dirichlet distribution parameter (for non-iid)
        """
        self.dataset_name = dataset_name.lower()
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.data_root = data_root
        self.distribution = distribution
        self.alpha = alpha
        
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
    
    def _load_shakespeare(self) -> Tuple[Dataset, Dataset]:
        """加载Shakespeare数据集（示例实现）/ Load Shakespeare dataset (example)"""
        class ShakespeareDataset(Dataset):
            def __init__(self, train=True):
                self.data = torch.randn(1000 if train else 200, 80)
                self.targets = torch.randint(0, 80, (1000 if train else 200,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        return ShakespeareDataset(train=True), ShakespeareDataset(train=False)
    
    def _create_client_data_splits(self) -> None:
        """为每个客户端创建独立的训练集和测试集 / Create independent train/test sets"""
        if self.distribution == "iid":
            self._create_iid_splits()
        else:
            self._create_non_iid_splits()
    
    def _create_iid_splits(self) -> None:
        """创建IID数据分布 / Create IID data distribution"""
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
            if i == self.num_clients - 1:  # 最后一个客户端获取剩余所有数据
                train_end = self.num_train_samples
            self.client_train_indices[i] = train_indices[train_start:train_end]
            
            # 测试集分配 / Test set allocation
            test_start = i * test_per_client
            test_end = test_start + test_per_client
            if i == self.num_clients - 1:  # 最后一个客户端获取剩余所有数据
                test_end = self.num_test_samples
            self.client_test_indices[i] = test_indices[test_start:test_end]
            
            if i < 5:  # 打印前5个客户端的信息
                print(f"  Client {i}: Train={len(self.client_train_indices[i])}, "
                      f"Test={len(self.client_test_indices[i])}")
        
        if self.num_clients > 5:
            print(f"  ... (remaining {self.num_clients - 5} clients)")
    
    def _create_non_iid_splits(self) -> None:
        """创建Non-IID数据分布（使用Dirichlet分布）/ Create Non-IID distribution"""
        print(f"\nNon-IID Distribution / Non-IID分布 (α={self.alpha}):")
        
        # 获取训练集标签 / Get training labels
        if hasattr(self.train_dataset, 'targets'):
            train_labels = np.array(self.train_dataset.targets)
        else:
            train_labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        
        # 获取测试集标签 / Get test labels
        if hasattr(self.test_dataset, 'targets'):
            test_labels = np.array(self.test_dataset.targets)
        else:
            test_labels = np.array([self.test_dataset[i][1] for i in range(len(self.test_dataset))])
        
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
            
            if i < 5:  # 打印前5个客户端的详细信息
                train_labels_i = train_labels[self.client_train_indices[i]]
                unique, counts = np.unique(train_labels_i, return_counts=True)
                class_dist = dict(zip(unique.tolist(), counts.tolist()))
                print(f"  Client {i}: Train={len(self.client_train_indices[i])}, "
                      f"Test={len(self.client_test_indices[i])}, "
                      f"Classes={list(class_dist.keys())[:5]}...")
        
        if self.num_clients > 5:
            print(f"  ... (remaining {self.num_clients - 5} clients)")
    
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
            batch_size=self.batch_size * 2,  # 测试时可用更大批次
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
        
        # 获取标签分布 / Get label distribution
        if hasattr(self.train_dataset, 'targets'):
            train_labels = np.array(self.train_dataset.targets)[train_indices]
        else:
            train_labels = np.array([self.train_dataset[i][1] for i in train_indices])
        
        if hasattr(self.test_dataset, 'targets'):
            test_labels = np.array(self.test_dataset.targets)[test_indices]
        else:
            test_labels = np.array([self.test_dataset[i][1] for i in test_indices])
        
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
    
    def visualize_data_distribution(self, save_path: str = None) -> None:
        """可视化数据分布 / Visualize data distribution"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        # 可视化前10个客户端 / Visualize first 10 clients
        for i in range(min(10, self.num_clients)):
            info = self.get_client_data_info(i)
            label_dist = info['train_label_distribution']
            
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