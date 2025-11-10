"""
incentive/differentiated_model.py
差异化模型分发系统 / Differentiated Model Distribution System
基于贡献度的个性化模型分配 / Contribution-based Personalized Model Allocation
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np


class DifferentiatedModelDistributor:
    """
    差异化模型分发器 / Differentiated Model Distributor
    根据客户端贡献度分发不同质量的模型
    Distribute models of different quality based on client contributions
    """
    
    def __init__(self, base_model: nn.Module, device: torch.device):
        """
        初始化模型分发器 / Initialize model distributor
        
        Args:
            base_model: 基础模型 / Base model
            device: 计算设备 / Computing device
        """
        self.base_model = copy.deepcopy(base_model).to(device)
        self.device = device
        
        # 存储不同贡献度对应的模型 / Store models for different contribution levels
        self.personalized_models = {}
        
        # 贡献度阈值 / Contribution thresholds
        self.contribution_thresholds = {
            'diamond': 0.75,  # 钻石级阈值
            'gold': 0.50,      # 金级阈值
            'silver': 0.25,    # 银级阈值
            'bronze': 0.01     # 铜级阈值
        }
    
    def create_personalized_model(self, client_id: int, 
                                 client_contribution: float,
                                 client_update: Dict[str, torch.Tensor],
                                 all_updates: List[Tuple[int, float, Dict[str, torch.Tensor]]],
                                 membership_level: str) -> Dict[str, torch.Tensor]:
        """
        为客户端创建个性化模型 / Create personalized model for client
        贡献度越高，获得来自其他客户端的更新越多
        Higher contribution gets more updates from other clients
        
        Args:
            client_id: 客户端ID / Client ID
            client_contribution: 客户端贡献度 / Client contribution
            client_update: 客户端自身的模型更新 / Client's own model update
            all_updates: 所有客户端的更新列表[(id, contribution, update)] / All clients' updates
            membership_level: 客户端会员等级 / Client membership level
            
        Returns:
            个性化模型权重 / Personalized model weights
        """
        if client_contribution == 0:
            # 贡献度为0，只获得自己的本地模型 / Zero contribution gets only local model
            return client_update
        
        # 根据贡献度排序所有更新 / Sort all updates by contribution
        sorted_updates = sorted(all_updates, key=lambda x: x[1], reverse=True)
        
        # 计算客户端可以获取的更新数量 / Calculate number of updates client can get
        # 贡献度越高，可以获取的其他客户端更新越多
        # Higher contribution allows access to more client updates
        max_updates = len(sorted_updates)
        
        # 根据贡献度和会员等级计算可获取的更新比例
        # Calculate update ratio based on contribution and membership level
        level_multiplier = {
            'diamond': 1.0,   # 钻石级可获取100%的更新
            'gold': 0.8,      # 金级可获取80%的更新
            'silver': 0.6,    # 银级可获取60%的更新
            'bronze': 0.4     # 铜级可获取40%的更新
        }.get(membership_level, 0.3)
        
        # 贡献度直接影响可获取的更新数量
        # Contribution directly affects accessible updates
        contribution_ratio = client_contribution  # 贡献度已经是0-1的值
        
        # 综合考虑等级和贡献度 / Consider both level and contribution
        # 即使等级相同，贡献度不同也会获得不同质量的模型
        # Even with same level, different contributions get different model quality
        effective_ratio = contribution_ratio * 0.7 + level_multiplier * 0.3
        num_accessible_updates = max(1, int(max_updates * effective_ratio))
        
        # 获取可访问的更新 / Get accessible updates
        accessible_updates = sorted_updates[:num_accessible_updates]
        
        # 聚合可访问的更新 / Aggregate accessible updates
        aggregated_model = self._aggregate_updates(
            client_update=client_update,
            accessible_updates=accessible_updates,
            client_contribution=client_contribution,
            client_id=client_id
        )
        
        return aggregated_model
    
    def _aggregate_updates(self, client_update: Dict[str, torch.Tensor],
                          accessible_updates: List[Tuple[int, float, Dict[str, torch.Tensor]]],
                          client_contribution: float,
                          client_id: int) -> Dict[str, torch.Tensor]:
        """
        聚合可访问的更新 / Aggregate accessible updates
        使用加权平均，权重基于贡献度
        Use weighted average based on contributions
        
        Args:
            client_update: 客户端自身更新 / Client's own update
            accessible_updates: 可访问的其他更新 / Accessible other updates
            client_contribution: 客户端贡献度 / Client contribution
            client_id: 客户端ID / Client ID
            
        Returns:
            聚合后的模型权重 / Aggregated model weights
        """
        aggregated_weights = {}
        
        # 计算总权重 / Calculate total weight
        total_weight = client_contribution  # 自己的权重
        
        # 添加其他客户端的权重 / Add other clients' weights
        for other_id, other_contribution, _ in accessible_updates:
            if other_id != client_id:  # 排除自己
                total_weight += other_contribution
        
        if total_weight == 0:
            return client_update
        
        # 初始化聚合权重 / Initialize aggregated weights
        for key in client_update.keys():
            # 先添加自己的更新 / First add own update
            weight_factor = client_contribution / total_weight
            aggregated_weights[key] = client_update[key] * weight_factor
        
        # 添加其他客户端的更新 / Add other clients' updates
        for other_id, other_contribution, other_update in accessible_updates:
            if other_id != client_id:
                weight_factor = other_contribution / total_weight
                for key in other_update.keys():
                    if key in aggregated_weights:
                        aggregated_weights[key] += other_update[key] * weight_factor
        
        return aggregated_weights
    
    def calculate_model_quality_score(self, model_weights: Dict[str, torch.Tensor],
                                     test_loader: torch.utils.data.DataLoader) -> float:
        """
        计算模型质量分数 / Calculate model quality score
        
        Args:
            model_weights: 模型权重 / Model weights
            test_loader: 测试数据加载器 / Test data loader
            
        Returns:
            模型质量分数(0-1) / Model quality score (0-1)
        """
        # 创建临时模型进行评估 / Create temporary model for evaluation
        temp_model = copy.deepcopy(self.base_model)
        temp_model.load_state_dict(model_weights)
        temp_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = temp_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def get_contribution_level(self, contribution: float) -> str:
        """
        根据贡献度获取等级 / Get level based on contribution
        
        Args:
            contribution: 贡献度值 / Contribution value
            
        Returns:
            贡献等级 / Contribution level
        """
        if contribution >= self.contribution_thresholds['diamond']:
            return 'diamond'
        elif contribution >= self.contribution_thresholds['gold']:
            return 'gold'
        elif contribution >= self.contribution_thresholds['silver']:
            return 'silver'
        elif contribution >= self.contribution_thresholds['bronze']:
            return 'bronze'
        else:
            return 'none'