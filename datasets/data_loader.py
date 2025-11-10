"""
数据集加载和预处理模块 / Dataset Loading and Preprocessing Module
支持MNIST, Fashion-MNIST, CIFAR-10/100, Shakespeare等数据集
Supports MNIST, Fashion-MNIST, CIFAR-10/100, Shakespeare and other datasets
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
        
        # 创建客户端数据索引 / Create client data indices
        self.client_indices = self._create_client_indices()
        
    def _load_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        加载指定的数据集 / Load the specified dataset
        
        Returns:
            训练集和测试集 / Training set and test set
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
        # 注意：实际应用中需要下载和处理Shakespeare数据集
        # Note: In actual use, you need to download and process the Shakespeare dataset
        
        class ShakespeareDataset(Dataset):
            """简化的Shakespeare数据集类 / Simplified Shakespeare dataset class"""
            
            def __init__(self, train=True):
                # 这里应该加载实际的Shakespeare文本数据
                # Here should load actual Shakespeare text data
                self.data = torch.randn(1000, 80)  # 示例数据 / Example data
                self.targets = torch.randint(0, 80, (1000,))  # 示例标签 / Example labels
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        return ShakespeareDataset(train=True), ShakespeareDataset(train=False)
    
    def _create_client_indices(self) -> Dict[int, List[int]]:
        """
        创建客户端数据索引 / Create client data indices
        
        Returns:
            客户端ID到数据索引的映射 / Mapping from client ID to data indices
        """
        if self.distribution == "iid":
            return self._create_iid_indices()
        else:
            return self._create_non_iid_indices()
    
    def _create_iid_indices(self) -> Dict[int, List[int]]:
        """
        创建IID数据分布的索引 / Create indices for IID data distribution
        """
        all_indices = list(range(self.num_train_samples))
        random.shuffle(all_indices)
        
        # 平均分配数据 / Evenly distribute data
        samples_per_client = self.num_train_samples // self.num_clients
        client_indices = {}
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            if i == self.num_clients - 1:
                # 最后一个客户端获取剩余的所有数据 / Last client gets all remaining data
                end_idx = self.num_train_samples
            client_indices[i] = all_indices[start_idx:end_idx]
        
        return client_indices
    
    def _create_non_iid_indices(self) -> Dict[int, List[int]]:
        """
        创建Non-IID数据分布的索引（使用Dirichlet分布） 
        Create indices for Non-IID data distribution (using Dirichlet distribution)
        """
        # 获取标签 / Get labels
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)
        else:
            labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        
        num_classes = len(np.unique(labels))
        
        # 使用Dirichlet分布生成客户端数据分配 / Use Dirichlet distribution to generate client data allocation
        client_indices = {i: [] for i in range(self.num_clients)}
        
        for k in range(num_classes):
            # 获取类别k的所有样本索引 / Get all sample indices for class k
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            
            # 使用Dirichlet分布分配样本 / Allocate samples using Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 分割索引并分配给客户端 / Split indices and assign to clients
            idx_splits = np.split(idx_k, proportions)
            for i, idx_split in enumerate(idx_splits):
                client_indices[i].extend(idx_split.tolist())
        
        return client_indices
    
    def get_client_dataloader(self, client_id: int) -> DataLoader:
        """
        获取指定客户端的数据加载器 / Get data loader for specified client
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            客户端的数据加载器 / Client's data loader
        """
        if client_id not in self.client_indices:
            raise ValueError(f"Invalid client ID: {client_id}")
        
        indices = self.client_indices[client_id]
        sampler = SubsetRandomSampler(indices)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,  # 避免多进程问题 / Avoid multiprocessing issues
            pin_memory=True
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """
        获取测试数据加载器 / Get test data loader
        
        Returns:
            测试数据加载器 / Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,  # 测试时可以使用更大的批次 / Can use larger batch for testing
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def get_client_data_info(self, client_id: int) -> Dict:
        """
        获取客户端数据信息 / Get client data information
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            包含数据量、类别分布等信息的字典 / Dictionary containing data size, class distribution, etc.
        """
        indices = self.client_indices[client_id]
        
        # 获取标签分布 / Get label distribution
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)[indices]
        else:
            labels = np.array([self.train_dataset[i][1] for i in indices])
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        return {
            'num_samples': len(indices),
            'label_distribution': dict(zip(unique_labels.tolist(), counts.tolist())),
            'data_indices': indices
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
            label_dist = info['label_distribution']
            
            axes[i].bar(label_dist.keys(), label_dist.values())
            axes[i].set_title(f'Client {i} (n={info["num_samples"]})')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
        
        plt.suptitle(f'{self.dataset_name} - {self.distribution.upper()} Distribution')
        plt.tight_layout()
        plt.savefig(f'data_distribution_{self.dataset_name}_{self.distribution}.png')
        plt.show()