"""
federated/server.py - 联邦学习服务器（重构版）
Federated Learning Server (Refactored)

基于层级约束动态梯度奖励的差异化分发
Differentiated distribution based on Tier-Constrained Dynamic Gradient Reward

核心流程 / Core Workflow:
1. 收集客户端梯度 / Collect client gradients
2. 聚合得到全局梯度 / Aggregate to global gradient
3. 计算CGSV贡献度 / Calculate CGSV contributions
4. 使用层级约束进行差异化稀疏 / Apply tier-constrained differential sparsification
5. 分发稀疏梯度给客户端 / Distribute sparse gradients to clients

修复说明 / Bug Fix:
- 修复了client_previous_weights更新时机的问题
- 原来在collect_client_updates中更新，导致梯度计算基准点错误
- 现在在客户端应用稀疏梯度后才更新，确保下一轮梯度计算正确
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incentive.sparsification_distributor import TierConstrainedGradientDistributor
from incentive.points_calculator import CGSVCalculator
from config import IncentiveConfig


class FederatedServerWithGradientSparsification:
    """
    联邦学习服务器 - 层级约束动态梯度奖励版本
    Federated Server - Tier-Constrained Dynamic Gradient Reward Version
    
    核心改进 / Core Improvements:
    1. 使用层级作为稀疏率的上下界 / Use tiers as bounds for keep ratios
    2. 组内插值实现连续映射 / Intra-tier interpolation for continuous mapping
    3. 基于幅度的稀疏化保留最重要参数 / Magnitude-based pruning to retain important parameters
    """
    
    def __init__(self, 
                 model, 
                 device,
                 tier_config: str = "default",
                 aggregation_method: str = "contribution"):
        """
        初始化服务器
        Initialize server
        
        Args:
            model: 全局模型 / Global model
            device: 计算设备 / Computing device
            tier_config: 层级配置 / Tier configuration
                - "default": 默认配置
                - "aggressive": 更激进的差异化
                - "moderate": 更温和的差异化
            aggregation_method: 聚合方式 / Aggregation method
                - "fedavg": 基于样本数量的FedAvg
                - "contribution": 基于贡献度的加权聚合
        """
        self.device = device
        self.global_model = copy.deepcopy(model).to(device)
        self.aggregation_method = aggregation_method
        
        # 获取全局模型权重（用于初始化）/ Get global model weights (for initialization)
        self.global_model_weights = {
            name: param.data.clone().detach()
            for name, param in self.global_model.named_parameters()
        }
        
        # 客户端状态追踪 / Client state tracking
        self.client_gradients = {}  # 存储客户端梯度 / Store client gradients
        self.client_train_info = {}  # 存储训练信息 / Store training info
        self.client_num_samples = {}  # 存储样本数量 / Store sample counts
        
        # 全局聚合梯度 / Global aggregated gradient
        self.aggregated_gradient = None
        
        # 客户端上一轮的模型权重（用于梯度计算）
        # Client previous weights (for gradient calculation)
        # 重要：这里存储的应该是客户端应用稀疏梯度后的状态
        # Important: This should store the state after client applies sparse gradient
        self.client_previous_weights = {}
        
        # 层级约束动态梯度分发器（核心改进）
        # Tier-Constrained Dynamic Gradient Distributor (core improvement)
        self.sparsification_distributor = TierConstrainedGradientDistributor(
            device=device,
            tier_config=tier_config,
            verbose=True
        )
        
        # 贡献度计算器 / Contribution calculator
        self.contribution_calculator = CGSVCalculator()
        
        # 统计信息 / Statistics
        self.round_contributions = {}
        self.round_stats = defaultdict(list)
        
        # 移动平均权重（用于平滑贡献度估计，参考论文公式4）
        # Moving average weights (for smoothing contribution estimates, ref: paper eq. 4)
        self.client_importance_coefficients = {}  # r_{i,t}
        self.moving_average_alpha = IncentiveConfig.MOVING_AVERAGE_ALPHA
        
        print(f"\n{'='*70}")
        print(f"Federated Server Initialized")
        print(f"联邦学习服务器已初始化")
        print(f"{'='*70}")
        print(f"Configuration / 配置:")
        print(f"  Tier Config / 层级配置: {tier_config}")
        print(f"  Aggregation Method / 聚合方式: {aggregation_method}")
        print(f"  Moving Average Alpha / 移动平均α: {self.moving_average_alpha}")
        print(f"{'='*70}\n")
    
    def reset_round(self):
        """重置轮次状态 / Reset round state"""
        self.client_gradients.clear()
        self.client_train_info.clear()
        self.client_num_samples.clear()
        self.aggregated_gradient = None
    
    def collect_client_updates(self, 
                               client_id: int,
                               updated_weights: Dict[str, torch.Tensor],
                               train_info: Dict):
        """
        收集客户端更新并计算梯度
        Collect client updates and calculate gradients
        
        核心：计算梯度 Δw_i = w_i^new - w_i^old
        Core: Calculate gradient Δw_i = w_i^new - w_i^old
        
        Args:
            client_id: 客户端ID / Client ID
            updated_weights: 客户端训练后的模型权重 / Updated model weights
            train_info: 训练信息 / Training info
        """
        # 获取客户端上一轮的权重 / Get client's previous weights
        if client_id not in self.client_previous_weights:
            # 第一轮：使用全局初始模型作为参考
            # First round: use global initial model as reference
            previous_weights = self.global_model_weights
        else:
            previous_weights = self.client_previous_weights[client_id]
        
        # 计算梯度：Δw = w_new - w_old
        # Calculate gradient: Δw = w_new - w_old
        gradient = {}
        for name in updated_weights.keys():
            gradient[name] = updated_weights[name].to(self.device) - \
                           previous_weights[name].to(self.device)
        
        # 存储梯度和训练信息 / Store gradient and training info
        self.client_gradients[client_id] = gradient
        self.client_train_info[client_id] = train_info
        self.client_num_samples[client_id] = train_info.get('num_samples', 1)
        
        # 注意：不在这里更新client_previous_weights！
        # 应该在客户端应用稀疏梯度后再更新，以确保梯度计算基准点正确
        # NOTE: Don't update client_previous_weights here!
        # Should update after client applies sparse gradient to ensure correct gradient baseline
    
    def update_global_model(self):
        """
        更新全局模型 - 聚合梯度
        Update global model - Aggregate gradients
        
        支持两种聚合方式 / Support two aggregation methods:
        1. FedAvg: 基于样本数量的加权平均
        2. Contribution: 基于贡献度的加权平均（参考论文）
        
        公式 / Formula:
        - FedAvg: Δw_global = Σ(n_k/n * Δw_k)
        - Contribution: Δw_global = Σ(s_k/Σs * Δw_k) where s_k is contribution
        """
        if not self.client_gradients:
            print("Warning: No client gradients to aggregate")
            return
        
        # 初始化聚合梯度 / Initialize aggregated gradient
        self.aggregated_gradient = {}
        sample_gradient = next(iter(self.client_gradients.values()))
        
        for name in sample_gradient.keys():
            self.aggregated_gradient[name] = torch.zeros_like(
                sample_gradient[name], 
                device=self.device
            )
        
        if self.aggregation_method == "contribution":
            # 基于贡献度的加权聚合 / Contribution-aware weighted aggregation
            # 先计算贡献度 / First calculate contributions
            contributions = self._calculate_instant_contributions()
            
            # 使用Softmax归一化贡献度 / Normalize contributions using Softmax
            if contributions:
                contrib_values = np.array(list(contributions.values()))
                # 使用温度参数控制区分度 / Use temperature to control discrimination
                scale = IncentiveConfig.AGGREGATION_SCALE
                exp_values = np.exp(scale * (contrib_values - np.max(contrib_values)))
                softmax_weights = exp_values / np.sum(exp_values)
                
                weight_dict = {cid: w for cid, w in zip(contributions.keys(), softmax_weights)}
            else:
                # 如果没有贡献度，退回到FedAvg / Fallback to FedAvg
                total_samples = sum(self.client_num_samples.values())
                weight_dict = {cid: n/total_samples for cid, n in self.client_num_samples.items()}
        else:
            # FedAvg: 基于样本数量 / FedAvg: sample-based
            total_samples = sum(self.client_num_samples.values())
            weight_dict = {cid: n/total_samples for cid, n in self.client_num_samples.items()}
        
        # 加权聚合 / Weighted aggregation
        for client_id, gradient in self.client_gradients.items():
            weight = weight_dict.get(client_id, 1.0 / len(self.client_gradients))
            
            for name in gradient.keys():
                self.aggregated_gradient[name] += weight * gradient[name].to(self.device)
        
        # 更新全局模型：w_global = w_global + Δw_global
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in self.aggregated_gradient:
                    param.data += self.aggregated_gradient[name]
        
        # 更新全局模型权重字典 / Update global model weights dict
        self.global_model_weights = {
            name: param.data.clone().detach()
            for name, param in self.global_model.named_parameters()
        }
    
    def _calculate_instant_contributions(self) -> Dict[int, float]:
        """
        计算瞬时贡献度（用于聚合权重）
        Calculate instant contributions (for aggregation weights)
        
        使用简化的余弦相似度 / Use simplified cosine similarity
        
        Returns:
            contributions: {client_id: contribution_score}
        """
        if not self.client_gradients:
            return {}
        
        # 计算所有梯度的平均作为参考 / Calculate mean of all gradients as reference
        avg_gradient = {}
        for name in next(iter(self.client_gradients.values())).keys():
            grads = [g[name] for g in self.client_gradients.values()]
            avg_gradient[name] = torch.stack(grads).mean(dim=0)
        
        contributions = {}
        for client_id, gradient in self.client_gradients.items():
            # 计算与平均梯度的余弦相似度 / Calculate cosine similarity with average gradient
            flat_client = torch.cat([g.flatten() for g in gradient.values()])
            flat_avg = torch.cat([g.flatten() for g in avg_gradient.values()])
            
            cos_sim = torch.nn.functional.cosine_similarity(
                flat_client.unsqueeze(0), flat_avg.unsqueeze(0)
            ).item()
            
            # 归一化到 [0, 1] / Normalize to [0, 1]
            contributions[client_id] = (cos_sim + 1.0) / 2.0
        
        return contributions
    
    def calculate_all_contributions(self, round_num: int) -> Dict[int, float]:
        """
        计算所有客户端的贡献度（CGSV）
        Calculate contributions for all clients (CGSV)
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            contributions: {client_id: contribution_score}
        """
        if not self.client_gradients:
            return {}
        
        # 使用CGSV计算器 / Use CGSV calculator
        contributions = self.contribution_calculator.calculate_all_contributions(
            client_gradients=self.client_gradients,
            aggregated_gradient=self.aggregated_gradient,
            client_num_samples=self.client_num_samples
        )
        
        # 更新移动平均（参考论文公式4）
        # Update moving average (reference: paper equation 4)
        # r_{i,t} = α * r_{i,t-1} + (1-α) * ψ_{i,t}
        for client_id, contrib in contributions.items():
            if client_id not in self.client_importance_coefficients:
                self.client_importance_coefficients[client_id] = contrib
            else:
                self.client_importance_coefficients[client_id] = (
                    self.moving_average_alpha * self.client_importance_coefficients[client_id] +
                    (1 - self.moving_average_alpha) * contrib
                )
        
        # 存储贡献度 / Store contributions
        self.round_contributions[round_num] = contributions.copy()
        self.round_contributions['current'] = contributions.copy()
        
        # 统计信息 / Statistics
        if contributions:
            contrib_values = list(contributions.values())
            self.round_stats['contributions_mean'].append(np.mean(contrib_values))
            self.round_stats['contributions_std'].append(np.std(contrib_values))
            self.round_stats['contributions_min'].append(np.min(contrib_values))
            self.round_stats['contributions_max'].append(np.max(contrib_values))
        
        return contributions
    
    def distribute_sparsified_gradients(self, 
                                        membership_levels: Dict[int, str]) -> Dict[int, Dict]:
        """
        分发稀疏化的梯度给客户端
        Distribute sparsified gradients to clients
        
        使用层级约束动态梯度奖励机制
        Using Tier-Constrained Dynamic Gradient Reward mechanism
        
        Args:
            membership_levels: 客户端会员等级 {client_id: level}
            
        Returns:
            Dict[client_id, sparse_gradient]: 每个客户端的稀疏化梯度
        """
        if self.aggregated_gradient is None:
            raise RuntimeError("Must call update_global_model() before distributing gradients")
        
        # 获取当前贡献度 / Get current contributions
        contributions = self.round_contributions.get('current', {})
        
        # 使用层级约束梯度分发器 / Use tier-constrained gradient distributor
        sparsified_gradients = self.sparsification_distributor.distribute_sparsified_gradients(
            global_gradient=self.aggregated_gradient,
            membership_levels=membership_levels,
            contributions=contributions
        )
        
        return sparsified_gradients
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """获取全局模型权重 / Get global model weights"""
        return copy.deepcopy(self.global_model_weights)
    
    def update_client_previous_weights(self, 
                                       client_id: int, 
                                       weights: Dict[str, torch.Tensor]):
        """
        更新客户端的"上一轮"权重记录
        Update client's "previous round" weights record
        
        应该在客户端应用稀疏梯度后调用，以确保下一轮梯度计算的基准点正确
        Should be called after client applies sparse gradient to ensure correct
        gradient baseline for next round
        
        核心修复 / Core Fix:
        - 原来在collect_client_updates中更新，导致基准点是训练后、应用稀疏梯度前的状态
        - 现在在客户端应用稀疏梯度后调用此方法更新，基准点是应用稀疏梯度后的状态
        
        Args:
            client_id: 客户端ID / Client ID
            weights: 客户端当前的模型权重（应用稀疏梯度后）
                     Client's current model weights (after applying sparse gradient)
        """
        self.client_previous_weights[client_id] = {
            name: param.clone().detach().to(self.device)
            for name, param in weights.items()
        }
    
    def get_round_summary(self, round_num: int) -> Dict:
        """获取轮次摘要信息 / Get round summary"""
        contributions = self.round_contributions.get(round_num, {})
        
        summary = {
            'round': round_num,
            'num_clients': len(self.client_gradients),
            'aggregation_method': self.aggregation_method,
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
        """获取贡献度统计信息 / Get contribution statistics"""
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
        """打印贡献度摘要 / Print contribution summary"""
        stats = self.get_contribution_statistics()
        if stats:
            print(f"\n{'='*60}")
            print(f"Contribution Statistics (CGSV)")
            print(f"贡献度统计（CGSV）")
            print(f"{'='*60}")
            print(f"Mean / 均值: {stats['mean']:.4f}")
            print(f"Std / 标准差: {stats['std']:.4f}")
            print(f"Range / 范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"Median / 中位数: {stats['median']:.4f}")
            print(f"{'='*60}")
    
    def print_sparsification_summary(self):
        """打印稀疏化摘要 / Print sparsification summary"""
        self.sparsification_distributor.print_round_summary()