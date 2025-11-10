"""
federated/client.py
客户端类定义 / Client Class Definition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import time
from typing import Dict, Optional, Tuple
import numpy as np


class FederatedClient:
    """
    联邦学习客户端 / Federated Learning Client
    负责本地训练和模型更新 / Responsible for local training and model updates
    """
    
    def __init__(self, client_id: int, model: nn.Module, dataloader: DataLoader,
                 device: torch.device = torch.device("cpu")):
        """
        初始化客户端 / Initialize client
        
        Args:
            client_id: 客户端ID / Client ID
            model: 模型 / Model
            dataloader: 数据加载器 / Data loader
            device: 计算设备 / Computing device
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = dataloader
        self.device = device
        
        # 训练统计 / Training statistics
        self.train_loss_history = []
        self.train_time = 0
        self.num_samples = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
        
        # 积分相关信息 / Points-related information
        self.total_points = 0
        self.membership_level = "bronze"
        self.participation_rounds = []
        
    def train(self, global_weights: Dict, epochs: int = 5, 
              lr: float = 0.01) -> Tuple[Dict, Dict]:
        """
        本地训练 / Local training
        
        Args:
            global_weights: 全局模型权重 / Global model weights
            epochs: 本地训练轮次 / Local training epochs
            lr: 学习率 / Learning rate
            
        Returns:
            更新的模型权重和训练信息 / Updated model weights and training information
        """
        # 加载全局模型权重 / Load global model weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        # 设置优化器 / Set optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 记录训练开始时间 / Record training start time
        start_time = time.time()
        
        total_loss = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                total_samples += len(data)
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            total_loss += avg_epoch_loss
        
        # 计算训练时间 / Calculate training time
        self.train_time = time.time() - start_time
        
        # 计算平均损失 / Calculate average loss
        avg_loss = total_loss / epochs if epochs > 0 else 0
        self.train_loss_history.append(avg_loss)
        
        # 准备训练信息 / Prepare training information
        train_info = {
            'client_id': self.client_id,
            'num_samples': total_samples,
            'train_time': self.train_time,
            'avg_loss': avg_loss,
            'data_size': self.num_samples,
            'computation_time': self.train_time,
            'model_quality': 1.0 / (1.0 + avg_loss)  # 简单的质量评估 / Simple quality assessment
        }
        
        return self.model.state_dict(), train_info
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        评估模型 / Evaluate model
        
        Args:
            test_loader: 测试数据加载器 / Test data loader
            
        Returns:
            准确率和损失 / Accuracy and loss
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        return accuracy, avg_loss
    
    def update_participation(self, round_num: int, points: float, level: str):
        """
        更新参与信息 / Update participation information
        
        Args:
            round_num: 轮次号 / Round number
            points: 获得的积分 / Points earned
            level: 新的会员等级 / New membership level
        """
        self.participation_rounds.append(round_num)
        self.total_points = points
        self.membership_level = level