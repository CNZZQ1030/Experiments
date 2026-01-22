"""
federated/client.py - 联邦学习客户端
Federated Learning Client

支持梯度累积更新，与层级约束动态梯度奖励机制配合
Support gradient accumulation update, compatible with Tier-Constrained Dynamic Gradient Reward

核心操作 / Core Operation:
    w_local^(t+1) = w_local^(t) + lr * sparse(Δw_global)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from typing import Dict, Tuple, Optional


class FederatedClient:
    """
    联邦学习客户端 - 梯度累积版本
    Federated Learning Client - Gradient Accumulation Version
    
    核心流程 / Core Workflow:
    1. 保持本地模型状态 / Maintain local model state
    2. 执行本地训练 / Perform local training
    3. 上传训练后的权重 / Upload trained weights
    4. 接收稀疏化梯度 / Receive sparsified gradients
    5. 应用梯度到本地模型 / Apply gradients to local model
    """
    
    def __init__(self, 
                 client_id: int,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 num_train_samples: int,
                 num_test_samples: int,
                 device: torch.device):
        """
        初始化客户端
        Initialize client
        
        Args:
            client_id: 客户端ID / Client ID
            model: 模型架构 / Model architecture
            train_dataloader: 训练数据加载器 / Training data loader
            test_dataloader: 测试数据加载器 / Test data loader
            num_train_samples: 训练样本数 / Number of training samples
            num_test_samples: 测试样本数 / Number of test samples
            device: 计算设备 / Computing device
        """
        self.client_id = client_id
        self.device = device
        
        # 创建本地模型（独立副本）/ Create local model (independent copy)
        self.local_model = copy.deepcopy(model).to(device)
        
        # 数据加载器 / Data loaders
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        
        # 损失函数 / Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # 会员等级 / Membership level
        self.membership_level = 'bronze'
        
        # 训练历史 / Training history
        self.train_history = {
            'loss': [],
            'accuracy': []
        }
        
        # 独立训练基准 / Standalone baseline
        self.standalone_accuracy = None
    
    def get_local_model_weights(self) -> Dict[str, torch.Tensor]:
        """获取本地模型权重 / Get local model weights"""
        return {
            name: param.data.clone().detach()
            for name, param in self.local_model.named_parameters()
        }
    
    def set_local_model_weights(self, weights: Dict[str, torch.Tensor]):
        """设置本地模型权重 / Set local model weights"""
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                if name in weights:
                    param.data = weights[name].to(self.device).clone()
    
    def apply_gradient_update(self, 
                             sparse_gradient: Dict[str, torch.Tensor], 
                             learning_rate: float = 1.0):
        """
        应用稀疏梯度到本地模型
        Apply sparse gradient to local model
        
        核心操作 / Core Operation:
            w_local = w_local + lr * sparse(Δw_global)
        
        这里的sparse(Δw_global)是经过层级约束稀疏化的全局梯度
        The sparse(Δw_global) is the tier-constrained sparsified global gradient
        
        Args:
            sparse_gradient: 稀疏化的全局梯度 / Sparsified global gradient
            learning_rate: 梯度应用的学习率 / Learning rate for gradient application
        """
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                if name in sparse_gradient:
                    # 应用稀疏梯度 / Apply sparse gradient
                    gradient_update = sparse_gradient[name].to(self.device)
                    param.data += learning_rate * gradient_update
    
    def train_federated(self, 
                       global_weights: Optional[Dict[str, torch.Tensor]] = None,
                       epochs: int = 1,
                       lr: float = 0.01) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        联邦学习本地训练
        Federated learning local training
        
        核心流程 / Core Workflow:
        1. 如果是第一轮，初始化本地模型为全局模型
        2. 否则，使用当前本地模型继续训练
        3. 训练后返回更新后的权重
        
        Args:
            global_weights: 全局模型权重（仅第一轮使用）/ Global model weights (first round only)
            epochs: 本地训练轮次 / Local training epochs
            lr: 学习率 / Learning rate
            
        Returns:
            updated_weights: 训练后的模型权重 / Updated model weights
            train_info: 训练信息 / Training information
        """
        # 第一轮：初始化本地模型 / First round: initialize local model
        if global_weights is not None and self.standalone_accuracy is None:
            self.set_local_model_weights(global_weights)
        
        # 设置为训练模式 / Set to training mode
        self.local_model.train()
        
        # 优化器 / Optimizer
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # 训练 / Training
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 统计 / Statistics
                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # 计算平均指标 / Calculate average metrics
        avg_loss = total_loss / (total_samples * epochs) if total_samples > 0 else 0
        avg_accuracy = total_correct / (total_samples * epochs) if total_samples > 0 else 0
        
        # 获取更新后的权重 / Get updated weights
        updated_weights = self.get_local_model_weights()
        
        # 测试准确率 / Test accuracy
        test_accuracy = self.evaluate()
        
        # 训练信息 / Training information
        train_info = {
            'client_id': self.client_id,
            'num_samples': self.num_train_samples,
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'federated_accuracy': test_accuracy,
            'membership_level': self.membership_level
        }
        
        # 记录历史 / Record history
        self.train_history['loss'].append(avg_loss)
        self.train_history['accuracy'].append(test_accuracy)
        
        return updated_weights, train_info
    
    def evaluate(self) -> float:
        """
        评估模型性能
        Evaluate model performance
        
        Returns:
            accuracy: 测试准确率 / Test accuracy
        """
        self.local_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.local_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def train_standalone(self, epochs: int = 20, lr: float = 0.01) -> Tuple[float, float]:
        """
        独立训练（用于基准对比）
        Standalone training (for baseline comparison)
        
        Args:
            epochs: 训练轮次 / Training epochs
            lr: 学习率 / Learning rate
            
        Returns:
            final_accuracy: 最终准确率 / Final accuracy
            best_accuracy: 最佳准确率 / Best accuracy
        """
        self.local_model.train()
        
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # 训练 / Training
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 评估 / Evaluate
            accuracy = self.evaluate()
            best_accuracy = max(best_accuracy, accuracy)
        
        final_accuracy = self.evaluate()
        self.standalone_accuracy = final_accuracy
        
        return final_accuracy, best_accuracy
    
    def update_membership_level(self, level: str):
        """更新会员等级 / Update membership level"""
        self.membership_level = level
    
    def get_membership_level(self) -> str:
        """获取会员等级 / Get membership level"""
        return self.membership_level
    
    def get_train_history(self) -> Dict:
        """获取训练历史 / Get training history"""
        return self.train_history
    
    def get_num_train_samples(self) -> int:
        """获取训练样本数 / Get number of training samples"""
        return self.num_train_samples
    
    def get_num_test_samples(self) -> int:
        """获取测试样本数 / Get number of test samples"""
        return self.num_test_samples