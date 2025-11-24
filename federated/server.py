"""
federated/server_sparsification.py
联邦学习服务器 - 集成稀疏化差异模型分发
Federated Learning Server - Integrated with Sparsification-based Distribution
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
from incentive.sparsification_distributor import SparsificationDistributor
from config import IncentiveConfig


class FederatedServerWithSparsification:
    """
    联邦学习服务器 - 稀疏化版本 / Federated Learning Server - Sparsification Version
    
    核心流程 / Core Workflow:
    1. 收集客户端更新 / Collect client updates
    2. 计算贡献度 / Calculate contributions
    3. 聚合更新得到全局模型 / Aggregate to get global model
    4. 基于贡献度对全局模型进行差异化稀疏 / Apply differentiated sparsification based on contributions
    5. 分发稀疏化模型给客户端 / Distribute sparsified models to clients
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
        
        # 稀疏化分发器 / Sparsification distributor
        self.sparsification_distributor = SparsificationDistributor(
            base_model=model,
            device=device,
            sparsity_ranges=IncentiveConfig.LEVEL_SPARSITY_RANGES,
            min_keep_ratio=IncentiveConfig.MIN_KEEP_RATIO,
            lambda_coefficient=IncentiveConfig.LAMBDA,
            sparsification_mode=IncentiveConfig.SPARSIFICATION_MODE
        )
        
        # 存储客户端信息 / Store client information
        self.client_contributions = {}  # 瞬时贡献 s_{i,t}
        self.client_gradients = {}      # 客户端梯度
        self.client_updates = {}        # 客户端模型权重
        self.client_levels = {}         # 客户端等级
        self.client_num_samples = {}    # 客户端样本数
        
        # 训练历史 / Training history
        self.round_history = []
        self.sparsification_history = []
        
        print(f"FederatedServerWithSparsification initialized")
        print(f"  Sparsification Mode: {IncentiveConfig.SPARSIFICATION_MODE}")
        print(f"  Lambda: {IncentiveConfig.LAMBDA}")
        print(f"  Min Keep Ratio: {IncentiveConfig.MIN_KEEP_RATIO}")
        
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
        # 计算梯度 g_{i,t} = θ_{new} - θ_{old}
        gradient = {}
        global_weights = self.global_model.state_dict()
        
        for key in client_weights.keys():
            gradient[key] = client_weights[key] - global_weights[key]
        
        self.client_gradients[client_id] = gradient
        self.client_updates[client_id] = client_weights
        self.client_num_samples[client_id] = client_info.get('num_samples', 1)
        self.client_levels[client_id] = client_info.get('membership_level', 'bronze')
    
    def calculate_all_contributions(self, round_num: int) -> Dict[int, float]:
        """
        计算所有客户端的瞬时贡献 / Calculate instantaneous contributions for all clients
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            贡献度字典 {client_id: s_{i,t}}
        """
        # 计算聚合梯度 / Calculate aggregated gradient
        aggregated_gradient = self._calculate_aggregated_gradient()
        
        # 使用CGSV计算贡献度 / Calculate contributions using CGSV
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
        使用FedAvg加权平均 / Use FedAvg weighted average
        
        Returns:
            聚合梯度 / Aggregated gradient
        """
        aggregated = {}
        total_samples = sum(self.client_num_samples.values())
        
        if total_samples == 0:
            return aggregated
        
        # 初始化 / Initialize
        for key in next(iter(self.client_gradients.values())).keys():
            sample_tensor = next(iter(self.client_gradients.values()))[key]
            aggregated[key] = torch.zeros_like(sample_tensor, dtype=torch.float32)
        
        # 加权平均 / Weighted average
        for client_id, gradient in self.client_gradients.items():
            weight = self.client_num_samples[client_id] / total_samples
            for key in gradient.keys():
                aggregated[key] += gradient[key].float() * weight
        
        return aggregated
    
    def update_global_model(self) -> None:
        """
        更新全局模型 / Update global model
        使用FedAvg算法进行聚合 / Use FedAvg for aggregation
        """
        if not self.client_updates:
            return
        
        # 计算总样本数 / Calculate total samples
        total_samples = sum(self.client_num_samples.values())
        
        if total_samples == 0:
            # 如果没有样本信息，使用均匀权重 / Use uniform weights if no sample info
            total_samples = len(self.client_updates)
            for client_id in self.client_updates:
                self.client_num_samples[client_id] = 1
        
        # 初始化聚合权重 / Initialize aggregated weights
        aggregated_weights = {}
        
        # FedAvg聚合 / FedAvg aggregation
        for client_id, client_weights in self.client_updates.items():
            weight_factor = self.client_num_samples[client_id] / total_samples
            
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
        
        print(f"Global model updated using FedAvg (total samples: {total_samples})")
    
    def distribute_sparsified_models(self,
                                    client_levels: Dict[int, str]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        基于稀疏化分发差异化模型 / Distribute differentiated models using sparsification
        
        核心思想：对全局模型进行不同程度的稀疏化
        Core idea: Apply different levels of sparsification to global model
        
        Args:
            client_levels: 客户端等级 {client_id: level}
            
        Returns:
            稀疏化模型字典 {client_id: sparsified_weights}
        """
        # 获取全局模型权重 / Get global model weights
        global_weights = self.global_model.state_dict()
        
        # 使用稀疏化分发器创建差异化模型 / Create differentiated models using sparsification
        sparsified_models = self.sparsification_distributor.distribute_all_sparsified_models(
            client_levels=client_levels,
            client_contributions=self.client_contributions,
            global_model_weights=global_weights
        )
        
        # 记录统计信息 / Record statistics
        stats = self.sparsification_distributor.get_sparsification_statistics()
        self.sparsification_history.append(stats)
        
        return sparsified_models
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """获取全局模型权重 / Get global model weights"""
        return self.global_model.state_dict()
    
    def reset_round(self) -> None:
        """重置轮次数据 / Reset round data"""
        self.client_gradients.clear()
        self.client_updates.clear()
        self.client_contributions.clear()
        self.client_num_samples.clear()
    
    def get_contribution_statistics(self) -> Dict:
        """获取贡献度统计信息 / Get contribution statistics"""
        contrib_stats = self.cgsv_calculator.get_contribution_statistics()
        
        # 添加稀疏化统计 / Add sparsification statistics
        if self.sparsification_history:
            latest_sparse_stats = self.sparsification_history[-1]
            contrib_stats['sparsification'] = latest_sparse_stats
        
        return contrib_stats
    
    def print_contribution_summary(self):
        """打印贡献度和稀疏化摘要 / Print contribution and sparsification summary"""
        # 打印CGSV贡献度统计 / Print CGSV contribution statistics
        self.cgsv_calculator.print_summary()
        
        # 打印稀疏化统计 / Print sparsification statistics
        self.sparsification_distributor.print_sparsification_summary()
    
    def get_round_summary(self, round_num: int) -> Dict:
        """
        获取轮次摘要 / Get round summary
        
        Args:
            round_num: 轮次号 / Round number
            
        Returns:
            轮次摘要信息 / Round summary information
        """
        sparse_stats = self.sparsification_distributor.get_sparsification_statistics()
        
        summary = {
            'round': round_num,
            'num_clients': len(self.client_updates),
            'contribution_stats': {
                'mean': np.mean(list(self.client_contributions.values())) if self.client_contributions else 0,
                'std': np.std(list(self.client_contributions.values())) if self.client_contributions else 0,
                'min': min(self.client_contributions.values()) if self.client_contributions else 0,
                'max': max(self.client_contributions.values()) if self.client_contributions else 0
            },
            'sparsification_stats': sparse_stats
        }
        
        return summary