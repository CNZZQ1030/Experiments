"""
federated/server.py
联邦学习服务器（支持差异化模型分发）/ Federated Learning Server (with Differentiated Model Distribution)
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incentive.points_calculator import AMACContributionCalculator
from incentive.differentiated_model import DifferentiatedModelDistributor


class FederatedServer:
    """
    联邦学习服务器 / Federated Learning Server
    支持基于AMAC贡献度的差异化模型分发
    Supports differentiated model distribution based on AMAC contribution
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")):
        """
        初始化服务器 / Initialize server
        
        Args:
            model: 全局模型 / Global model
            device: 计算设备 / Computing device
        """
        self.global_model = copy.deepcopy(model).to(device)
        self.device = device
        
        # AMAC贡献度计算器 / AMAC contribution calculator
        self.amac_calculator = AMACContributionCalculator(T=200, gamma=1.0)
        
        # 差异化模型分发器 / Differentiated model distributor
        self.model_distributor = DifferentiatedModelDistributor(model, device)
        
        # 存储客户端信息 / Store client information
        self.client_contributions = {}  # 客户端贡献度
        self.client_gradients = {}      # 客户端梯度
        self.client_updates = {}        # 客户端模型更新
        self.client_levels = {}         # 客户端等级
        
        # 训练历史 / Training history
        self.round_history = []
        self.contribution_history = []
        
    def collect_client_updates(self, client_id: int,
                              client_weights: Dict[str, torch.Tensor],
                              client_info: Dict) -> None:
        """
        收集客户端更新 / Collect client updates
        
        Args:
            client_id: 客户端ID / Client ID
            client_weights: 客户端模型权重 / Client model weights
            client_info: 客户端信息 / Client information
        """
        # 计算梯度（更新量）/ Calculate gradient (update amount)
        gradient = {}
        global_weights = self.global_model.state_dict()
        
        for key in client_weights.keys():
            gradient[key] = client_weights[key] - global_weights[key]
        
        self.client_gradients[client_id] = gradient
        self.client_updates[client_id] = client_weights
        
        # 存储客户端等级信息 / Store client level information
        self.client_levels[client_id] = client_info.get('membership_level', 'bronze')
    
    def calculate_all_contributions(self, round_num: int) -> Dict[int, float]:
        """
        计算所有客户端的AMAC贡献度 / Calculate AMAC contributions for all clients
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            客户端贡献度字典 / Client contribution dictionary
        """
        # 首先计算聚合梯度 / First calculate aggregated gradient
        aggregated_gradient = self._calculate_aggregated_gradient()
        
        # 获取所有客户端梯度列表 / Get all client gradients list
        all_gradients = list(self.client_gradients.values())
        
        # 计算每个客户端的贡献度 / Calculate contribution for each client
        contributions = {}
        for client_id, client_gradient in self.client_gradients.items():
            contribution = self.amac_calculator.calculate_contribution(
                client_id=client_id,
                round_num=round_num,
                client_gradient=client_gradient,
                aggregated_gradient=aggregated_gradient,
                all_gradients=all_gradients
            )
            contributions[client_id] = contribution
        
        self.client_contributions = contributions
        return contributions
    
    def _calculate_aggregated_gradient(self) -> Dict[str, torch.Tensor]:
        """
        计算聚合梯度 / Calculate aggregated gradient
        
        Returns:
            聚合梯度 / Aggregated gradient
        """
        aggregated = {}
        num_clients = len(self.client_gradients)
        
        if num_clients == 0:
            return aggregated
        
        # 初始化聚合梯度 / Initialize aggregated gradient
        for key in next(iter(self.client_gradients.values())).keys():
            aggregated[key] = torch.zeros_like(
                next(iter(self.client_gradients.values()))[key]
            )
        
        # 简单平均所有梯度 / Simple average of all gradients
        for gradient in self.client_gradients.values():
            for key in gradient.keys():
                aggregated[key] += gradient[key] / num_clients
        
        return aggregated
    
    def distribute_personalized_models(self, round_num: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        分发个性化模型 / Distribute personalized models
        基于贡献度为每个客户端创建差异化模型
        Create differentiated models for each client based on contribution
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            个性化模型字典 / Personalized models dictionary
        """
        # 计算所有客户端贡献度 / Calculate all client contributions
        contributions = self.calculate_all_contributions(round_num)
        
        # 准备所有更新列表 / Prepare all updates list
        all_updates = []
        for client_id, contribution in contributions.items():
            all_updates.append((
                client_id,
                contribution,
                self.client_updates[client_id]
            ))
        
        # 为每个客户端创建个性化模型 / Create personalized model for each client
        personalized_models = {}
        for client_id in self.client_updates.keys():
            client_contribution = contributions.get(client_id, 0)
            client_level = self.client_levels.get(client_id, 'bronze')
            
            personalized_model = self.model_distributor.create_personalized_model(
                client_id=client_id,
                client_contribution=client_contribution,
                client_update=self.client_updates[client_id],
                all_updates=all_updates,
                membership_level=client_level
            )
            
            personalized_models[client_id] = personalized_model
        
        return personalized_models
    
    def update_global_model(self) -> None:
        """
        更新全局模型 / Update global model
        使用基于贡献度的加权聚合
        Use contribution-based weighted aggregation
        """
        if not self.client_updates:
            return
        
        # 获取贡献度总和 / Get total contribution
        total_contribution = sum(self.client_contributions.values())
        if total_contribution == 0:
            total_contribution = len(self.client_contributions)  # 退化为平均
        
        # 初始化聚合权重 / Initialize aggregated weights
        aggregated_weights = {}
        
        for client_id, client_weights in self.client_updates.items():
            contribution = self.client_contributions.get(client_id, 1.0 / len(self.client_updates))
            weight_factor = contribution / total_contribution
            
            for key in client_weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(client_weights[key])
                aggregated_weights[key] += client_weights[key] * weight_factor
        
        # 更新全局模型 / Update global model
        self.global_model.load_state_dict(aggregated_weights)
    
    def evaluate_model(self, model_weights: Dict[str, torch.Tensor],
                      test_loader: DataLoader) -> Tuple[float, float]:
        """
        评估模型 / Evaluate model
        
        Args:
            model_weights: 模型权重 / Model weights
            test_loader: 测试数据加载器 / Test data loader
            
        Returns:
            (准确率, 损失) / (accuracy, loss)
        """
        # 创建临时模型 / Create temporary model
        temp_model = copy.deepcopy(self.global_model)
        temp_model.load_state_dict(model_weights)
        temp_model.eval()
        
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = temp_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        return accuracy, avg_loss
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取全局模型权重 / Get global model weights
        
        Returns:
            全局模型权重 / Global model weights
        """
        return self.global_model.state_dict()
    
    def reset_round(self) -> None:
        """
        重置轮次数据 / Reset round data
        清空当前轮次的客户端更新和梯度
        Clear current round's client updates and gradients
        """
        self.client_gradients.clear()
        self.client_updates.clear()
    
    def get_contribution_statistics(self) -> Dict:
        """
        获取贡献度统计信息 / Get contribution statistics
        
        Returns:
            统计信息字典 / Statistics dictionary
        """
        if not self.client_contributions:
            return {}
        
        contributions = list(self.client_contributions.values())
        return {
            'mean': np.mean(contributions),
            'std': np.std(contributions),
            'min': np.min(contributions),
            'max': np.max(contributions),
            'num_clients': len(contributions)
        }