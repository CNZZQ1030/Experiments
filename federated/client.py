"""
federated/client.py
联邦学习客户端（使用本地测试集）/ Federated Learning Client (with Local Test Set)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import time
from typing import Dict, Optional, Tuple


class FederatedClient:
    """
    联邦学习客户端 / Federated Learning Client
    使用本地测试集进行评估，支持独立训练基准计算
    Uses local test set for evaluation, supports standalone training baseline
    """
    
    def __init__(self, client_id: int, model: nn.Module, 
                 train_dataloader: DataLoader, test_dataloader: DataLoader,
                 device: torch.device = torch.device("cpu")):
        """
        初始化客户端 / Initialize client
        
        Args:
            client_id: 客户端ID / Client ID
            model: 模型 / Model
            train_dataloader: 训练数据加载器（客户端本地训练集）/ Training data loader (client's local train set)
            test_dataloader: 测试数据加载器（客户端本地测试集）/ Test data loader (client's local test set)
            device: 计算设备 / Computing device
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader  # 客户端自己的测试集 / Client's own test set
        self.device = device
        
        # 独立训练基准 / Standalone training baseline
        self.standalone_accuracy = None
        self.standalone_loss = None
        self.standalone_model = None
        
        # 联邦学习性能 / Federated learning performance
        self.federated_accuracy = None
        self.federated_loss = None
        
        # 训练统计 / Training statistics
        self.training_time = 0
        self.num_train_samples = len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else 0
        self.num_test_samples = len(test_dataloader.dataset) if hasattr(test_dataloader, 'dataset') else 0
        
        # 会员等级 / Membership level
        self.membership_level = "bronze"
        
        # 贡献度历史 / Contribution history
        self.contribution_history = []
        
        print(f"Client {client_id} initialized: Train samples={self.num_train_samples}, Test samples={self.num_test_samples}")
    
    def train_standalone(self, epochs: int = 10, lr: float = 0.01) -> Tuple[float, float]:
        """
        独立训练（不参与联邦学习）/ Standalone training (without federated learning)
        在客户端自己的本地测试集上评估
        Evaluated on client's own local test set
        
        Args:
            epochs: 训练轮次 / Training epochs
            lr: 学习率 / Learning rate
            
        Returns:
            (独立准确率, 独立损失) / (standalone accuracy, standalone loss)
        """
        # 创建独立模型副本 / Create standalone model copy
        self.standalone_model = copy.deepcopy(self.model)
        self.standalone_model.train()
        
        optimizer = optim.SGD(self.standalone_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 训练独立模型 / Train standalone model
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.standalone_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # 在本地测试集上评估独立模型 / Evaluate standalone model on local test set
        self.standalone_accuracy, self.standalone_loss = self._evaluate_model(
            self.standalone_model, self.test_dataloader
        )
        
        print(f"Client {self.client_id} - Standalone Accuracy (on local test set): {self.standalone_accuracy:.4f}")
        
        return self.standalone_accuracy, self.standalone_loss
    
    def train_federated(self, global_weights: Dict[str, torch.Tensor],
                       epochs: int = 5, lr: float = 0.01) -> Tuple[Dict, Dict]:
        """
        联邦学习训练 / Federated learning training
        在客户端自己的本地测试集上评估
        Evaluated on client's own local test set
        
        Args:
            global_weights: 全局模型权重（可能是个性化的）/ Global model weights (possibly personalized)
            epochs: 本地训练轮次 / Local training epochs
            lr: 学习率 / Learning rate
            
        Returns:
            (更新后的模型权重, 训练信息) / (updated model weights, training info)
        """
        # 加载全局模型权重 / Load global model weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 记录训练开始时间 / Record training start time
        start_time = time.time()
        
        total_loss = 0
        batch_count = 0
        
        # 本地训练 / Local training
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
        
        # 计算训练时间 / Calculate training time
        self.training_time = time.time() - start_time
        
        # 在本地测试集上评估联邦学习模型 / Evaluate federated model on local test set
        self.federated_accuracy, self.federated_loss = self._evaluate_model(
            self.model, self.test_dataloader
        )
        
        # 准备训练信息 / Prepare training info
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        train_info = {
            'client_id': self.client_id,
            'num_samples': self.num_train_samples,
            'training_time': self.training_time,
            'avg_loss': avg_loss,
            'federated_accuracy': self.federated_accuracy,
            'federated_loss': self.federated_loss,
            'membership_level': self.membership_level,
            'model_quality': 1.0 / (1.0 + avg_loss),
        }
        
        return self.model.state_dict(), train_info
    
    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """
        评估模型（在指定的数据加载器上）/ Evaluate model (on specified data loader)
        
        Args:
            model: 要评估的模型 / Model to evaluate
            dataloader: 数据加载器（本地测试集）/ Data loader (local test set)
            
        Returns:
            (准确率, 损失) / (accuracy, loss)
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        return accuracy, avg_loss
    
    def update_membership_level(self, new_level: str) -> None:
        """
        更新会员等级 / Update membership level
        
        Args:
            new_level: 新等级 / New level
        """
        self.membership_level = new_level
    
    def get_performance_improvement(self) -> float:
        """
        获取性能改进 / Get performance improvement
        计算联邦学习相对于独立训练的改进（在本地测试集上）
        Calculate improvement of federated learning over standalone training (on local test set)
        
        Returns:
            性能改进百分比 / Performance improvement percentage
        """
        if self.standalone_accuracy is None or self.federated_accuracy is None:
            return 0.0
        
        if self.standalone_accuracy == 0:
            return 100.0 if self.federated_accuracy > 0 else 0.0
        
        improvement = ((self.federated_accuracy - self.standalone_accuracy) 
                      / self.standalone_accuracy) * 100
        return improvement
    
    def get_client_metrics(self) -> Dict:
        """
        获取客户端指标 / Get client metrics
        
        Returns:
            客户端指标字典 / Client metrics dictionary
        """
        return {
            'client_id': self.client_id,
            'standalone_accuracy': self.standalone_accuracy,
            'federated_accuracy': self.federated_accuracy,
            'performance_improvement': self.get_performance_improvement(),
            'membership_level': self.membership_level,
            'num_train_samples': self.num_train_samples,
            'num_test_samples': self.num_test_samples,
            'training_time': self.training_time
        }