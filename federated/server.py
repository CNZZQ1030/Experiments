"""
federated/server.py (Updated for UPSM Implementation)
联邦学习服务器 - 集成UPSM统一概率采样机制
Federated Learning Server - Integrated with UPSM

实现PDF Algorithm 1的服务器端逻辑 / Implements server-side logic from PDF Algorithm 1
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incentive.points_calculator import CGSVContributionCalculator
from incentive.differentiated_model import UPSMDistributor
from config import IncentiveConfig


class FederatedServer:
    """
    联邦学习服务器 / Federated Learning Server
    
    实现PDF文档中的完整工作流程 / Implements complete workflow from PDF:
    1. 计算瞬时贡献 / Calculate instantaneous contribution
    2. 基于UPSM的差异化模型分发 / UPSM-based differentiated model distribution
    3. 贡献度加权的全局模型更新 / Contribution-weighted global model update
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
        
        # CGSV贡献度计算器 / CGSV contribution calculator
        self.cgsv_calculator = CGSVContributionCalculator(
            epsilon=IncentiveConfig.CGSV_EPSILON
        )
        
        # UPSM分发器 / UPSM distributor
        self.upsm_distributor = UPSMDistributor(
            base_model=model,
            device=device,
            access_ratios=IncentiveConfig.LEVEL_ACCESS_RATIOS,
            selection_bias=IncentiveConfig.LEVEL_SELECTION_BIAS
        )
        
        # 存储客户端信息 / Store client information
        self.client_contributions = {}  # 瞬时贡献 s_{i,t}
        self.client_gradients = {}      # 客户端梯度
        self.client_updates = {}        # 客户端模型权重
        self.client_levels = {}         # 客户端等级
        
        # 训练历史 / Training history
        self.round_history = []
        
        print(f"FederatedServer initialized with UPSM mechanism")
        
    def collect_client_updates(self, client_id: int,
                              client_weights: Dict[str, torch.Tensor],
                              client_info: Dict) -> None:
        """
        收集客户端更新 / Collect client updates
        
        计算梯度（模型更新量）/ Calculate gradient (model update)
        g_{i,t} = θ_{new} - θ_{old}
        
        Args:
            client_id: 客户端ID / Client ID
            client_weights: 客户端模型权重 / Client model weights
            client_info: 客户端信息 / Client information
        """
        # 计算梯度 / Calculate gradient
        gradient = {}
        global_weights = self.global_model.state_dict()
        
        for key in client_weights.keys():
            gradient[key] = client_weights[key] - global_weights[key]
        
        self.client_gradients[client_id] = gradient
        self.client_updates[client_id] = client_weights
        
        # 存储等级信息 / Store level information
        self.client_levels[client_id] = client_info.get('membership_level', 'bronze')
    
    def calculate_all_contributions(self, round_num: int) -> Dict[int, float]:
        """
        计算所有客户端的瞬时贡献 / Calculate instantaneous contributions for all clients
        
        实现PDF Algorithm 1 步骤3-6 / Implements PDF Algorithm 1 steps 3-6
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            贡献度字典 {client_id: s_{i,t}}
        """
        # 计算聚合梯度 g_{agg} = Average({g_{i,t}})
        aggregated_gradient = self._calculate_aggregated_gradient()
        
        # 计算所有客户端的CGSV贡献
        contributions = self.cgsv_calculator.calculate_all_contributions(
            round_num=round_num,
            client_gradients=self.client_gradients,
            aggregated_gradient=aggregated_gradient
        )
        
        self.client_contributions = contributions
        return contributions
    
    def _calculate_aggregated_gradient(self) -> Dict[str, torch.Tensor]:
        """
        计算聚合梯度 / Calculate aggregated gradient
        
        g_{agg} = (1/N) Σ g_{i,t}
        
        Returns:
            聚合梯度 / Aggregated gradient
        """
        aggregated = {}
        num_clients = len(self.client_gradients)
        
        if num_clients == 0:
            return aggregated
        
        # 初始化 / Initialize
        for key in next(iter(self.client_gradients.values())).keys():
            sample_tensor = next(iter(self.client_gradients.values()))[key]
            aggregated[key] = torch.zeros_like(sample_tensor, dtype=torch.float32)
        
        # 平均所有梯度 / Average all gradients
        for gradient in self.client_gradients.values():
            for key in gradient.keys():
                aggregated[key] += gradient[key].float() / num_clients
        
        return aggregated
    
    def distribute_personalized_models(self, 
                                      client_levels: Dict[int, str]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        基于UPSM分发个性化模型 / Distribute personalized models using UPSM
        
        实现PDF Algorithm 1 步骤12-18 / Implements PDF Algorithm 1 steps 12-18
        
        Args:
            client_levels: 客户端等级 {client_id: level}
            
        Returns:
            个性化模型字典 {client_id: weights}
        """
        # 更新等级信息 / Update level information
        self.client_levels = client_levels
        
        # 使用UPSM分发 / Distribute using UPSM
        personalized_models = self.upsm_distributor.distribute_all_personalized_models(
            client_levels=client_levels,
            all_contributions=self.client_contributions,
            all_updates=self.client_updates
        )
        
        return personalized_models
    
    def update_global_model(self) -> None:
        """
        更新全局模型 / Update global model
        
        使用贡献度加权聚合：
        θ_{global} = Σ (s_{i,t} / Σ s_{k,t}) * θ_i
        """
        if not self.client_updates:
            return
        
        # 计算贡献度总和 / Calculate total contribution
        total_contribution = sum(self.client_contributions.values())
        
        if total_contribution < 1e-10:
            # 如果总贡献为0，使用均匀权重 / Use uniform weights if total is 0
            total_contribution = len(self.client_contributions)
            weights_dict = {cid: 1.0 for cid in self.client_contributions}
        else:
            weights_dict = self.client_contributions
        
        # 初始化聚合权重 / Initialize aggregated weights
        aggregated_weights = {}
        
        for client_id, client_weights in self.client_updates.items():
            weight_factor = weights_dict.get(client_id, 1.0 / len(self.client_updates)) / total_contribution
            
            for key in client_weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(client_weights[key], dtype=torch.float32)
                
                if client_weights[key].dtype in [torch.int32, torch.int64, torch.long]:
                    aggregated_weights[key] += client_weights[key].float() * weight_factor
                else:
                    aggregated_weights[key] += client_weights[key] * weight_factor
        
        # 转换回原始类型 / Convert back to original dtype
        for key in aggregated_weights.keys():
            sample_weight = next(iter(self.client_updates.values()))[key]
            if sample_weight.dtype in [torch.int32, torch.int64, torch.long]:
                aggregated_weights[key] = aggregated_weights[key].round().to(sample_weight.dtype)
        
        # 更新全局模型 / Update global model
        self.global_model.load_state_dict(aggregated_weights)
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """获取全局模型权重 / Get global model weights"""
        return self.global_model.state_dict()
    
    def reset_round(self) -> None:
        """重置轮次数据 / Reset round data"""
        self.client_gradients.clear()
        self.client_updates.clear()
        self.client_contributions.clear()
    
    def get_contribution_statistics(self) -> Dict:
        """获取贡献度统计信息 / Get contribution statistics"""
        return self.cgsv_calculator.get_contribution_statistics()
    
    def print_contribution_summary(self):
        """打印贡献度摘要 / Print contribution summary"""
        self.cgsv_calculator.print_summary()
        
        # 打印UPSM统计 / Print UPSM statistics
        if self.client_levels:
            stats = self.upsm_distributor.get_sampling_statistics(
                self.client_levels, len(self.client_levels)
            )
            
            print(f"\n{'='*70}")
            print(f"UPSM Sampling Statistics")
            print(f"{'='*70}")
            
            for level, level_stats in stats['level_stats'].items():
                print(f"\n{level.capitalize()}:")
                print(f"  Count: {level_stats['count']}")
                print(f"  Access Ratio (ρ): {level_stats['access_ratio']}")
                print(f"  Accessible Updates (K): {level_stats['num_accessible']}")
                print(f"  Selection Bias (β): {level_stats['selection_bias']}")
                print(f"  Strategy: {level_stats['sampling_strategy']}")
            
            print(f"{'='*70}")