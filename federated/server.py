"""
federated/server.py - 重构版本
联邦学习服务器 - 基于梯度稀疏化的差异化分发
Federated Server - Gradient Sparsification-based Differentiated Distribution
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incentive.sparsification_distributor import SparsificationDistributor
from incentive.points_calculator import CGSVCalculator


class FederatedServerWithGradientSparsification:
    """
    联邦学习服务器 - 梯度稀疏化版本
    
    核心流程 / Core Workflow:
    1. 收集客户端梯度 / Collect client gradients
    2. 聚合得到全局梯度 / Aggregate to global gradient  
    3. 对全局梯度进行差异化稀疏 / Sparsify global gradient differentially
    4. 分发稀疏梯度给客户端 / Distribute sparse gradients to clients
    5. 客户端应用到本地模型 / Clients apply to local models
    """
    
    def __init__(self, model, device):
        """
        初始化服务器
        
        Args:
            model: 全局模型
            device: 计算设备
        """
        self.device = device
        self.global_model = copy.deepcopy(model).to(device)
        
        # 获取全局模型权重（用于初始化）
        self.global_model_weights = {
            name: param.data.clone().detach()
            for name, param in self.global_model.named_parameters()
        }
        
        # 客户端状态追踪
        self.client_gradients = {}  # 存储客户端梯度
        self.client_train_info = {}  # 存储训练信息
        self.client_num_samples = {}  # 存储样本数量
        
        # 全局聚合梯度
        self.aggregated_gradient = None
        
        # 客户端上一轮的模型权重（用于梯度计算）
        self.client_previous_weights = {}
        
        # 稀疏化分发器
        self.sparsification_distributor = SparsificationDistributor(device)
        
        # 贡献度计算器
        self.contribution_calculator = CGSVCalculator()
        
        # 统计信息
        self.round_contributions = {}
        self.round_stats = defaultdict(list)
        
        print(f"✓ Server initialized (Gradient Sparsification Mode)")
    
    def reset_round(self):
        """重置轮次状态"""
        self.client_gradients.clear()
        self.client_train_info.clear()
        self.client_num_samples.clear()
        self.aggregated_gradient = None
    
    def collect_client_updates(self, client_id: int, 
                              updated_weights: Dict[str, torch.Tensor],
                              train_info: Dict):
        """
        收集客户端更新并计算梯度
        
        核心改变：不再直接存储模型权重，而是计算并存储梯度
        
        Args:
            client_id: 客户端ID
            updated_weights: 客户端训练后的模型权重
            train_info: 训练信息（损失、准确率等）
        """
        # 获取客户端上一轮的权重
        if client_id not in self.client_previous_weights:
            # 第一轮：使用全局初始模型作为参考
            previous_weights = self.global_model_weights
        else:
            previous_weights = self.client_previous_weights[client_id]
        
        # 计算梯度：Δw = w_new - w_old
        gradient = {}
        for name in updated_weights.keys():
            gradient[name] = updated_weights[name].to(self.device) - \
                           previous_weights[name].to(self.device)
        
        # 存储梯度和训练信息
        self.client_gradients[client_id] = gradient
        self.client_train_info[client_id] = train_info
        self.client_num_samples[client_id] = train_info.get('num_samples', 1)
        
        # 更新客户端的上一轮权重（用于下一轮计算梯度）
        self.client_previous_weights[client_id] = {
            name: param.clone().detach()
            for name, param in updated_weights.items()
        }
    
    def update_global_model(self):
        """
        更新全局模型 - 使用FedAvg聚合梯度
        
        核心改变：聚合的是梯度，不是模型权重
        公式：Δw_global = Σ(n_k/n * Δw_k)
        然后：w_global = w_global + Δw_global
        """
        if not self.client_gradients:
            print("Warning: No client gradients to aggregate")
            return
        
        # 计算总样本数
        total_samples = sum(self.client_num_samples.values())
        
        # 初始化聚合梯度
        self.aggregated_gradient = {}
        sample_gradient = next(iter(self.client_gradients.values()))
        
        for name in sample_gradient.keys():
            self.aggregated_gradient[name] = torch.zeros_like(
                sample_gradient[name], 
                device=self.device
            )
        
        # FedAvg聚合：加权平均梯度
        for client_id, gradient in self.client_gradients.items():
            weight = self.client_num_samples[client_id] / total_samples
            
            for name in gradient.keys():
                self.aggregated_gradient[name] += weight * gradient[name].to(self.device)
        
        # 更新全局模型：w_global = w_global + Δw_global
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in self.aggregated_gradient:
                    param.data += self.aggregated_gradient[name]
        
        # 更新全局模型权重字典
        self.global_model_weights = {
            name: param.data.clone().detach()
            for name, param in self.global_model.named_parameters()
        }
    
    def distribute_sparsified_gradients(self, membership_levels: Dict[int, str]) -> Dict[int, Dict]:
        """
        分发稀疏化的梯度给客户端
        
        核心创新：对全局聚合梯度进行差异化稀疏，而不是对模型权重稀疏
        
        Args:
            membership_levels: 客户端会员等级 {client_id: level}
        
        Returns:
            Dict[client_id, sparse_gradient]: 每个客户端的稀疏化梯度
        """
        if self.aggregated_gradient is None:
            raise RuntimeError("Must call update_global_model() before distributing gradients")
        
        # 计算贡献度排名（用于确定稀疏率）
        contributions = self.round_contributions.get('current', {})
        
        # 使用稀疏化分发器对梯度进行差异化稀疏
        sparsified_gradients = self.sparsification_distributor.distribute_sparsified_gradients(
            global_gradient=self.aggregated_gradient,
            membership_levels=membership_levels,
            contributions=contributions
        )
        
        return sparsified_gradients
    
    def calculate_all_contributions(self, round_num: int) -> Dict[int, float]:
        """
        计算所有客户端的贡献度（CGSV）
        
        Args:
            round_num: 当前轮次
        
        Returns:
            Dict[client_id, contribution]: 贡献度字典
        """
        if not self.client_gradients:
            return {}
        
        contributions = self.contribution_calculator.calculate_all_contributions(
            client_gradients=self.client_gradients,
            aggregated_gradient=self.aggregated_gradient,
            client_num_samples=self.client_num_samples
        )
        
        # 存储贡献度
        self.round_contributions[round_num] = contributions
        self.round_contributions['current'] = contributions
        
        # 统计信息
        if contributions:
            contrib_values = list(contributions.values())
            self.round_stats['contributions_mean'].append(np.mean(contrib_values))
            self.round_stats['contributions_std'].append(np.std(contrib_values))
        
        return contributions
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """获取全局模型权重（用于客户端初始化）"""
        return copy.deepcopy(self.global_model_weights)
    
    def get_round_summary(self, round_num: int) -> Dict:
        """获取轮次摘要信息"""
        contributions = self.round_contributions.get(round_num, {})
        
        summary = {
            'round': round_num,
            'num_clients': len(self.client_gradients),
            'contribution_stats': {
                'mean': np.mean(list(contributions.values())) if contributions else 0,
                'std': np.std(list(contributions.values())) if contributions else 0,
                'min': np.min(list(contributions.values())) if contributions else 0,
                'max': np.max(list(contributions.values())) if contributions else 0
            },
            'sparsification_stats': self.sparsification_distributor.get_sparsification_statistics()
        }
        
        return summary
    
    def get_contribution_statistics(self) -> Dict:
        """获取贡献度统计信息"""
        all_contributions = []
        for round_num, contrib_dict in self.round_contributions.items():
            if round_num != 'current' and isinstance(contrib_dict, dict):
                all_contributions.extend(contrib_dict.values())
        
        if not all_contributions:
            return {}
        
        return {
            'mean': float(np.mean(all_contributions)),
            'std': float(np.std(all_contributions)),
            'min': float(np.min(all_contributions)),
            'max': float(np.max(all_contributions)),
            'median': float(np.median(all_contributions))
        }
    
    def print_contribution_summary(self):
        """打印贡献度摘要"""
        stats = self.get_contribution_statistics()
        if stats:
            print(f"\n{'='*80}")
            print("Contribution Statistics (CGSV)")
            print(f"{'='*80}")
            print(f"Mean: {stats['mean']:.4f}")
            print(f"Std:  {stats['std']:.4f}")
            print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"Median: {stats['median']:.4f}")
            print(f"{'='*80}")