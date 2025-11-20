"""
federated/client.py
联邦学习客户端 / Federated Learning Client
每个客户端拥有独立的训练集和测试集
Each client has independent training and test sets
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
    支持独立训练和联邦学习训练
    Supports standalone training and federated learning training
    """
    
    def __init__(self, client_id: int, model: nn.Module, 
                 train_dataloader: DataLoader, test_dataloader: DataLoader,
                 num_train_samples: int, num_test_samples: int,
                 device: torch.device = torch.device("cpu")):
        """
        初始化客户端 / Initialize client
        
        Args:
            client_id: 客户端ID / Client ID
            model: 模型 / Model
            train_dataloader: 训练数据加载器 / Training dataloader
            test_dataloader: 测试数据加载器 / Test dataloader
            num_train_samples: 训练样本数 / Number of training samples
            num_test_samples: 测试样本数 / Number of test samples
            device: 计算设备 / Computing device
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        
        # 样本数量 / Sample counts
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        
        # 独立训练性能 / Standalone performance
        self.standalone_accuracy = None
        self.standalone_loss = None
        self.standalone_model = None
        
        # 联邦学习性能 / Federated performance
        self.federated_accuracy = None
        self.federated_loss = None
        
        # 训练统计 / Training statistics
        self.training_time = 0
        
        # 会员等级 / Membership level
        self.membership_level = "bronze"
        
        # 贡献度历史 / Contribution history
        self.contribution_history = []
        
        if client_id < 5:  # 只打印前5个客户端
            print(f"Client {client_id} initialized: Train={self.num_train_samples}, Test={self.num_test_samples}")
    
    def train_standalone(self, epochs: int = 10, lr: float = 0.01) -> Tuple[float, float]:
        """
        独立训练（不参与联邦学习）/ Standalone training
        在客户端自己的测试集上评估
        Evaluated on client's own test set
        
        Args:
            epochs: 训练轮次 / Training epochs
            lr: 学习率 / Learning rate
            
        Returns:
            (独立准确率, 独立损失) / (standalone accuracy, standalone loss)
        """
        # 创建独立模型副本 / Create standalone model
        self.standalone_model = copy.deepcopy(self.model)
        self.standalone_model.train()
        
        optimizer = optim.SGD(self.standalone_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 训练 / Training
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.standalone_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # 在测试集上评估 / Evaluate on test set
        self.standalone_accuracy, self.standalone_loss = self._evaluate_model(
            self.standalone_model, self.test_dataloader
        )
        
        if self.client_id < 5:  # 只打印前5个客户端
            print(f"Client {self.client_id} - Standalone Accuracy: {self.standalone_accuracy:.4f}")
        
        return self.standalone_accuracy, self.standalone_loss
    
    def train_federated(self, global_weights: Dict[str, torch.Tensor],
                       epochs: int = 1, lr: float = 0.01) -> Tuple[Dict, Dict]:
        """
        联邦学习训练 / Federated learning training
        在客户端自己的测试集上评估
        Evaluated on client's own test set
        
        Args:
            global_weights: 全局模型权重（可能是个性化的）/ Global model weights
            epochs: 本地训练轮次 / Local training epochs
            lr: 学习率 / Learning rate
            
        Returns:
            (更新后的模型权重, 训练信息) / (updated weights, training info)
        """
        # 加载全局模型 / Load global model
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 记录训练时间 / Record training time
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
        
        # 在测试集上评估 / Evaluate on test set
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
        评估模型 / Evaluate model
        
        Args:
            model: 要评估的模型 / Model to evaluate
            dataloader: 数据加载器 / Dataloader
            
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
        """更新会员等级 / Update membership level"""
        self.membership_level = new_level
    
    def get_performance_improvement(self) -> float:
        """
        获取性能改进 / Get performance improvement
        计算联邦学习相对于独立训练的改进
        Calculate improvement of federated over standalone
        
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
    
    def get_absolute_improvement(self) -> float:
        """
        获取绝对性能改进 / Get absolute performance improvement
        
        Returns:
            绝对改进值 / Absolute improvement value
        """
        if self.standalone_accuracy is None or self.federated_accuracy is None:
            return 0.0
        
        return self.federated_accuracy - self.standalone_accuracy
    
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
            'absolute_improvement': self.get_absolute_improvement(),
            'membership_level': self.membership_level,
            'num_train_samples': self.num_train_samples,
            'num_test_samples': self.num_test_samples,
            'training_time': self.training_time
        }