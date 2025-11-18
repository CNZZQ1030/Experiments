"""
federated/server.py (Updated for CGSV with Relative Ranking)
联邦学习服务器 - 基于CGSV贡献度和相对排名的差异化模型分发
Federated Learning Server - Differentiated model distribution based on CGSV and relative ranking
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

from incentive.points_calculator import CGSVContributionCalculator
from incentive.differentiated_model import DifferentiatedModelDistributor


class FederatedServer:
    """
    联邦学习服务器 / Federated Learning Server
    
    核心改进 / Core Improvements:
    1. 使用CGSV代替AMAC / Use CGSV instead of AMAC
    2. 批量计算贡献度并归一化 / Batch calculate and normalize contributions
    3. 支持相对排名的会员系统 / Support membership system with relative ranking
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu"),
                 use_relative_normalization: bool = True):
        """
        初始化服务器 / Initialize server
        
        Args:
            model: 全局模型 / Global model
            device: 计算设备 / Computing device
            use_relative_normalization: 是否使用相对归一化 / Whether to use relative normalization
        """
        self.global_model = copy.deepcopy(model).to(device)
        self.device = device
        
        # CGSV贡献度计算器（增强版）/ CGSV contribution calculator (enhanced)
        self.cgsv_calculator = CGSVContributionCalculator(
            use_relative_normalization=use_relative_normalization,
            normalization_method='mean'  # 可选: 'mean', 'minmax', 'zscore'
        )
        
        # 差异化模型分发器 / Differentiated model distributor
        self.model_distributor = DifferentiatedModelDistributor(model, device)
        
        # 存储客户端信息 / Store client information
        self.client_contributions = {}  # 客户端贡献度（归一化后）
        self.client_raw_contributions = {}  # 原始贡献度
        self.client_gradients = {}  # 客户端梯度
        self.client_updates = {}  # 客户端模型更新
        self.client_levels = {}  # 客户端等级
        
        # 训练历史 / Training history
        self.round_history = []
        self.contribution_history = []
        
        print(f"FederatedServer initialized with CGSV contribution calculation")
        print(f"  Relative Normalization: {use_relative_normalization}")
        
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
    
    def calculate_all_contributions(self, round_num: int) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        计算所有客户端的CGSV贡献度（批量+归一化）
        Calculate CGSV contributions for all clients (batch + normalization)
        
        核心改进：使用批量计算方法获得更好的归一化效果
        Core improvement: Use batch calculation for better normalization
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            (归一化贡献度, 原始贡献度) / (normalized contributions, raw contributions)
        """
        # 首先计算聚合梯度 / First calculate aggregated gradient
        aggregated_gradient = self._calculate_aggregated_gradient()
        
        # 使用批量计算方法 / Use batch calculation method
        normalized_contributions = self.cgsv_calculator.calculate_contributions_batch(
            round_num=round_num,
            client_gradients=self.client_gradients,
            aggregated_gradient=aggregated_gradient
        )
        
        # 获取原始贡献度（用于分析）/ Get raw contributions (for analysis)
        raw_contributions = {}
        for client_id in self.client_gradients.keys():
            history = self.cgsv_calculator.get_client_contribution_history(client_id)
            if history:
                raw_contributions[client_id] = history[-1]['raw_contribution']
            else:
                raw_contributions[client_id] = 0.0
        
        # 保存结果 / Save results
        self.client_contributions = normalized_contributions
        self.client_raw_contributions = raw_contributions
        
        return normalized_contributions, raw_contributions
    
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
            sample_tensor = next(iter(self.client_gradients.values()))[key]
            aggregated[key] = torch.zeros_like(sample_tensor, dtype=sample_tensor.dtype)
        
        # 简单平均所有梯度 / Simple average of all gradients
        for gradient in self.client_gradients.values():
            for key in gradient.keys():
                if gradient[key].dtype in [torch.int32, torch.int64, torch.long]:
                    aggregated[key] = aggregated[key].float()
                    aggregated[key] += gradient[key].float() / num_clients
                else:
                    aggregated[key] += gradient[key] / num_clients
        
        return aggregated
    
    def distribute_personalized_models(self, round_num: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        分发个性化模型 / Distribute personalized models
        基于当前贡献度和会员等级
        Based on current contribution and membership level
        """
        # 计算所有客户端贡献度（归一化）
        normalized_contributions, raw_contributions = self.calculate_all_contributions(round_num)
        
        # 保存贡献度 / Save contributions
        self.client_contributions = normalized_contributions
        self.client_raw_contributions = raw_contributions
    
        # 准备所有更新列表 / Prepare all updates list
        all_updates = []
        for client_id, contribution in normalized_contributions.items():
            all_updates.append((
                client_id,
                contribution,  # 使用归一化后的贡献度
                self.client_updates[client_id]
            ))
        
        # 为每个客户端创建个性化模型 / Create personalized model for each client
        personalized_models = {}
        for client_id in self.client_updates.keys():
            client_contribution = normalized_contributions.get(client_id, 0)
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
        
        # 获取归一化贡献度总和 / Get total normalized contribution
        total_contribution = sum(self.client_contributions.values())
        if total_contribution < 1e-6:
            total_contribution = len(self.client_contributions)  # 退化为平均
        
        # 初始化聚合权重 / Initialize aggregated weights
        aggregated_weights = {}
        
        for client_id, client_weights in self.client_updates.items():
            contribution = self.client_contributions.get(client_id, 1.0 / len(self.client_updates))
            weight_factor = contribution / total_contribution
            
            for key in client_weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(client_weights[key])
                
                # 处理不同数据类型
                if client_weights[key].dtype in [torch.int32, torch.int64, torch.long]:
                    if aggregated_weights[key].dtype in [torch.int32, torch.int64, torch.long]:
                        aggregated_weights[key] = aggregated_weights[key].float()
                    aggregated_weights[key] += client_weights[key].float() * weight_factor
                else:
                    aggregated_weights[key] += client_weights[key] * weight_factor
        
        # 将整数类型的权重转回原类型
        for key in aggregated_weights.keys():
            sample_weight = next(iter(self.client_updates.values()))[key]
            if sample_weight.dtype in [torch.int32, torch.int64, torch.long]:
                aggregated_weights[key] = aggregated_weights[key].round().to(sample_weight.dtype)
        
        # 更新全局模型 / Update global model
        self.global_model.load_state_dict(aggregated_weights)
    
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
        
        normalized_values = list(self.client_contributions.values())
        raw_values = list(self.client_raw_contributions.values())
        
        return {
            'normalized': {
                'mean': np.mean(normalized_values),
                'std': np.std(normalized_values),
                'min': np.min(normalized_values),
                'max': np.max(normalized_values),
            },
            'raw': {
                'mean': np.mean(raw_values),
                'std': np.std(raw_values),
                'min': np.min(raw_values),
                'max': np.max(raw_values),
            },
            'num_clients': len(self.client_contributions)
        }
    
    def print_contribution_summary(self):
        """打印贡献度摘要 / Print contribution summary"""
        self.cgsv_calculator.print_summary()