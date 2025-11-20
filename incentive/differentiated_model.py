"""
incentive/differentiated_model.py (Rewritten for UPSM)
差异化模型分发系统 - 统一概率采样机制 / Differentiated Model Distribution System - UPSM

实现PDF文档Section 7.4中描述的UPSM机制 / Implements UPSM from PDF Section 7.4:
1. 数量控制 (Quantity Control) - 信息访问率 ρ_L
2. 质量控制 (Quality Control) - 选择偏差系数 β_L with Boltzmann分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np


class UPSMDistributor:
    """
    统一概率采样机制分发器 / Unified Probabilistic Sampling Mechanism Distributor
    
    核心功能 / Core Functions:
    1. 数量控制：根据等级确定可访问的更新数量 K_L
    2. 质量控制：使用Boltzmann分布采样高质量更新
    3. 最终聚合：贡献度加权聚合采样的更新
    """
    
    def __init__(self, base_model: nn.Module, device: torch.device,
                 access_ratios: Dict[str, float] = None,
                 selection_bias: Dict[str, float] = None):
        """
        初始化UPSM分发器 / Initialize UPSM distributor
        
        Args:
            base_model: 基础模型 / Base model
            device: 计算设备 / Computing device
            access_ratios: 信息访问率 ρ_L / Information access ratios
            selection_bias: 选择偏差系数 β_L / Selection bias coefficients
        """
        self.base_model = copy.deepcopy(base_model).to(device)
        self.device = device
        
        # 默认参数来自PDF Table 1 / Default parameters from PDF Table 1
        if access_ratios is None:
            self.access_ratios = {
                'diamond': 1.0,
                'gold': 0.8,
                'silver': 0.5,
                'bronze': 0.2
            }
        else:
            self.access_ratios = access_ratios
        
        # 默认参数来自PDF Table 2 / Default parameters from PDF Table 2
        if selection_bias is None:
            self.selection_bias = {
                'diamond': 10.0,
                'gold': 3.0,
                'silver': 1.0,
                'bronze': 0.0
            }
        else:
            self.selection_bias = selection_bias
        
        print(f"UPSMDistributor initialized:")
        print(f"  Access Ratios (ρ): {self.access_ratios}")
        print(f"  Selection Bias (β): {self.selection_bias}")
    
    def z_score_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Z-Score标准化贡献分数 / Z-Score normalize contribution scores
        
        公式: ŝ = (s - μ) / σ
        Formula: ŝ = (s - μ) / σ
        
        Args:
            scores: 原始贡献分数 / Raw contribution scores
            
        Returns:
            标准化后的分数 / Normalized scores
        """
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std < 1e-10:
            # 如果标准差为0，返回全0
            return np.zeros_like(scores)
        
        return (scores - mean) / std
    
    def boltzmann_sampling(self, standardized_scores: np.ndarray, 
                          beta: float, 
                          num_samples: int,
                          client_indices: List[int]) -> List[int]:
        """
        Boltzmann分布采样 / Boltzmann distribution sampling
        
        公式 / Formula: P_j = exp(β · ŝ_j) / Σ exp(β · ŝ_k)
        
        Args:
            standardized_scores: Z-Score标准化后的分数 / Z-Score normalized scores
            beta: 选择偏差系数 / Selection bias coefficient
            num_samples: 采样数量 K / Number of samples K
            client_indices: 客户端索引列表 / List of client indices
            
        Returns:
            采样的客户端索引列表 / List of sampled client indices
        """
        n = len(standardized_scores)
        
        if n == 0:
            return []
        
        num_samples = min(num_samples, n)
        
        if beta == 0:
            # β = 0: 均匀随机采样 / Uniform random sampling
            sampled_indices = np.random.choice(n, size=num_samples, replace=False)
        else:
            # 计算Boltzmann概率 / Calculate Boltzmann probabilities
            # 使用数值稳定的softmax / Use numerically stable softmax
            scaled_scores = beta * standardized_scores
            # 减去最大值防止溢出 / Subtract max to prevent overflow
            scaled_scores = scaled_scores - np.max(scaled_scores)
            exp_scores = np.exp(scaled_scores)
            probabilities = exp_scores / np.sum(exp_scores)
            
            # 多项式采样（无放回）/ Multinomial sampling (without replacement)
            sampled_indices = []
            remaining_indices = list(range(n))
            remaining_probs = probabilities.copy()
            
            for _ in range(num_samples):
                if len(remaining_indices) == 0:
                    break
                    
                # 归一化剩余概率 / Normalize remaining probabilities
                prob_sum = np.sum(remaining_probs)
                if prob_sum < 1e-10:
                    # 如果概率和太小，使用均匀采样
                    idx = np.random.choice(len(remaining_indices))
                else:
                    normalized_probs = remaining_probs / prob_sum
                    idx = np.random.choice(len(remaining_indices), p=normalized_probs)
                
                sampled_indices.append(remaining_indices[idx])
                
                # 移除已选择的元素 / Remove selected element
                remaining_indices.pop(idx)
                remaining_probs = np.delete(remaining_probs, idx)
        
        # 转换为客户端ID / Convert to client IDs
        return [client_indices[i] for i in sampled_indices]
    
    def create_personalized_model(self, 
                                 target_client_id: int,
                                 target_level: str,
                                 all_contributions: Dict[int, float],
                                 all_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        为目标客户端创建个性化模型 / Create personalized model for target client
        
        实现PDF Algorithm 1的步骤13-18 / Implements steps 13-18 of PDF Algorithm 1
        
        Args:
            target_client_id: 目标客户端ID / Target client ID
            target_level: 目标客户端等级 / Target client level
            all_contributions: 所有客户端的瞬时贡献 {client_id: s_{i,t}}
            all_updates: 所有客户端的模型更新 {client_id: weights}
            
        Returns:
            个性化模型权重 / Personalized model weights
        """
        N = len(all_contributions)
        
        if N == 0:
            return all_updates.get(target_client_id, {})
        
        # 准备数据 / Prepare data
        client_ids = list(all_contributions.keys())
        contributions = np.array([all_contributions[cid] for cid in client_ids])
        
        # ===== 步骤1: 数量控制 (Quantity Control) =====
        # K = max(1, floor(N × ρ_L))
        rho = self.access_ratios.get(target_level, 0.2)
        K = max(1, int(np.floor(N * rho)))
        
        # ===== 步骤2: 质量控制 (Quality Control) =====
        # Z-Score标准化 / Z-Score normalization
        standardized_scores = self.z_score_normalize(contributions)
        
        # 获取选择偏差系数 / Get selection bias coefficient
        beta = self.selection_bias.get(target_level, 0.0)
        
        # Boltzmann采样 / Boltzmann sampling
        sampled_client_ids = self.boltzmann_sampling(
            standardized_scores=standardized_scores,
            beta=beta,
            num_samples=K,
            client_indices=client_ids
        )
        
        # ===== 步骤3: 最终聚合 (Final Aggregation) =====
        # θ_{i,t+1} = θ_{i,t} - η Σ_{j∈S_i} (s_{j,t} / Σ_{k∈S_i} s_{k,t}) g_{j,t}
        # 注意：这里我们返回聚合后的模型权重，而不是梯度
        
        if not sampled_client_ids:
            return all_updates.get(target_client_id, {})
        
        # 计算采样集合的贡献度总和 / Calculate total contribution of sampled set
        sampled_contributions = {cid: all_contributions[cid] for cid in sampled_client_ids}
        total_contribution = sum(sampled_contributions.values())
        
        if total_contribution < 1e-10:
            # 如果总贡献为0，使用均匀权重 / If total is 0, use uniform weights
            total_contribution = len(sampled_client_ids)
            sampled_contributions = {cid: 1.0 for cid in sampled_client_ids}
        
        # 加权聚合 / Weighted aggregation
        aggregated_weights = {}
        
        for cid in sampled_client_ids:
            weight_factor = sampled_contributions[cid] / total_contribution
            client_update = all_updates[cid]
            
            for key in client_update.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(client_update[key], dtype=torch.float32)
                
                # 处理不同数据类型 / Handle different data types
                if client_update[key].dtype in [torch.int32, torch.int64, torch.long]:
                    aggregated_weights[key] += client_update[key].float() * weight_factor
                else:
                    aggregated_weights[key] += client_update[key] * weight_factor
        
        # 转换回原始数据类型 / Convert back to original dtype
        if sampled_client_ids:
            sample_update = all_updates[sampled_client_ids[0]]
            for key in aggregated_weights.keys():
                if sample_update[key].dtype in [torch.int32, torch.int64, torch.long]:
                    aggregated_weights[key] = aggregated_weights[key].round().to(sample_update[key].dtype)
        
        return aggregated_weights
    
    def distribute_all_personalized_models(self,
                                          client_levels: Dict[int, str],
                                          all_contributions: Dict[int, float],
                                          all_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        为所有客户端分发个性化模型 / Distribute personalized models to all clients
        
        Args:
            client_levels: 客户端等级 {client_id: level}
            all_contributions: 所有贡献 {client_id: contribution}
            all_updates: 所有更新 {client_id: weights}
            
        Returns:
            个性化模型字典 {client_id: personalized_weights}
        """
        personalized_models = {}
        
        for client_id in all_updates.keys():
            level = client_levels.get(client_id, 'bronze')
            
            personalized_model = self.create_personalized_model(
                target_client_id=client_id,
                target_level=level,
                all_contributions=all_contributions,
                all_updates=all_updates
            )
            
            personalized_models[client_id] = personalized_model
        
        return personalized_models
    
    def get_sampling_statistics(self, 
                               client_levels: Dict[int, str],
                               num_clients: int) -> Dict:
        """
        获取采样统计信息 / Get sampling statistics
        
        Args:
            client_levels: 客户端等级 / Client levels
            num_clients: 客户端总数 / Total number of clients
            
        Returns:
            统计信息 / Statistics
        """
        stats = {
            'num_clients': num_clients,
            'level_stats': {}
        }
        
        for level in ['diamond', 'gold', 'silver', 'bronze']:
            rho = self.access_ratios[level]
            beta = self.selection_bias[level]
            K = max(1, int(np.floor(num_clients * rho)))
            
            level_count = sum(1 for l in client_levels.values() if l == level)
            
            stats['level_stats'][level] = {
                'count': level_count,
                'access_ratio': rho,
                'num_accessible': K,
                'selection_bias': beta,
                'sampling_strategy': self._get_strategy_description(beta)
            }
        
        return stats
    
    def _get_strategy_description(self, beta: float) -> str:
        """获取采样策略描述 / Get sampling strategy description"""
        if beta >= 10.0:
            return "Deterministic Exploitation (确定性择优)"
        elif beta >= 3.0:
            return "Probabilistic Exploitation (概率性择优)"
        elif beta >= 1.0:
            return "Weak Preference (弱偏好)"
        else:
            return "Uniform Random (纯随机)"


# 为了向后兼容，保留原类名 / Keep original class name for backward compatibility
DifferentiatedModelDistributor = UPSMDistributor