"""
incentive/points_calculator.py - CGSV贡献度计算器
CGSV Contribution Calculator

基于NeurIPS 2021论文"Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"
Based on NeurIPS 2021 paper "Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"

核心公式 / Core Formula (论文公式3 / Paper Equation 3):
    CGSV_i ≈ ψ_i = cos(u_i, u_N)
    
其中 / Where:
    u_i = Γ * Δw_i / ||Δw_i||  (归一化的客户端梯度 / Normalized client gradient)
    u_N = Σ r_{i,t-1} * u_i     (聚合梯度 / Aggregated gradient)

CGSV用于 / CGSV is used for:
1. 计算客户端的瞬时贡献度 / Calculate instant contribution
2. 作为层级约束动态梯度奖励的输入 / Input for Tier-Constrained Dynamic Gradient Reward
3. 决定客户端在层级内的相对位置 / Determine client's relative position within tier
"""

import torch
import numpy as np
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IncentiveConfig


class CGSVCalculator:
    """
    Cosine Gradient Shapley Value (CGSV) 计算器
    Cosine Gradient Shapley Value Calculator
    
    计算客户端梯度与全局梯度的余弦相似度作为贡献度
    Calculate cosine similarity between client gradient and global gradient as contribution
    
    公式 / Formula:
        CGSV_i = cos(Δw_i, Δw_global) = (Δw_i · Δw_global) / (||Δw_i|| * ||Δw_global||)
    
    返回值范围 / Return range:
        原始余弦相似度: [-1, 1]
        归一化后: [0, 1]
    """
    
    def __init__(self):
        """初始化CGSV计算器 / Initialize CGSV calculator"""
        self.epsilon = IncentiveConfig.CGSV_EPSILON
    
    def flatten_gradient(self, gradient: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将梯度字典展平为一维张量
        Flatten gradient dictionary to 1D tensor
        
        Args:
            gradient: 梯度字典 {param_name: gradient_tensor}
            
        Returns:
            flat_gradient: 一维张量
        """
        flat_tensors = []
        for name in sorted(gradient.keys()):
            flat_tensors.append(gradient[name].flatten())
        
        return torch.cat(flat_tensors)
    
    def cosine_similarity(self, 
                         gradient_a: torch.Tensor, 
                         gradient_b: torch.Tensor) -> float:
        """
        计算两个梯度的余弦相似度
        Calculate cosine similarity between two gradients
        
        公式 / Formula:
            cos(a, b) = (a · b) / (||a|| * ||b||)
        
        Args:
            gradient_a: 梯度A (已展平) / Gradient A (flattened)
            gradient_b: 梯度B (已展平) / Gradient B (flattened)
            
        Returns:
            similarity: 余弦相似度 [-1, 1]
        """
        # 计算点积 / Calculate dot product
        dot_product = torch.dot(gradient_a, gradient_b)
        
        # 计算范数 / Calculate norms
        norm_a = torch.norm(gradient_a)
        norm_b = torch.norm(gradient_b)
        
        # 避免除零 / Avoid division by zero
        if norm_a < self.epsilon or norm_b < self.epsilon:
            return 0.0
        
        # 余弦相似度 / Cosine similarity
        similarity = dot_product / (norm_a * norm_b)
        
        return float(similarity.item())
    
    def calculate_contribution(self,
                              client_gradient: Dict[str, torch.Tensor],
                              aggregated_gradient: Dict[str, torch.Tensor]) -> float:
        """
        计算单个客户端的贡献度
        Calculate contribution for a single client
        
        这是论文公式(3)的近似实现
        This is the approximation of paper equation (3):
        ψ_i ≈ cos(u_i, u_N)
        
        Args:
            client_gradient: 客户端梯度 Δw_i
            aggregated_gradient: 全局聚合梯度 Δw_global
            
        Returns:
            contribution: 贡献度值 [0, 1]
        """
        # 展平梯度 / Flatten gradients
        flat_client = self.flatten_gradient(client_gradient)
        flat_global = self.flatten_gradient(aggregated_gradient)
        
        # 计算余弦相似度 / Calculate cosine similarity
        similarity = self.cosine_similarity(flat_client, flat_global)
        
        # 归一化到 [0, 1] / Normalize to [0, 1]
        # cos ∈ [-1, 1] → contribution ∈ [0, 1]
        contribution = (similarity + 1.0) / 2.0
        
        return contribution
    
    def calculate_all_contributions(self,
                                   client_gradients: Dict[int, Dict[str, torch.Tensor]],
                                   aggregated_gradient: Dict[str, torch.Tensor],
                                   client_num_samples: Dict[int, int]) -> Dict[int, float]:
        """
        计算所有客户端的贡献度
        Calculate contributions for all clients
        
        这个方法的输出将被用于:
        1. 层级分配 (assign_clients_to_tiers)
        2. 组内插值 (calculate_intra_tier_keep_ratio)
        3. 最终保留率计算 (calculate_all_keep_ratios)
        
        The output of this method will be used for:
        1. Tier assignment
        2. Intra-tier interpolation
        3. Final keep ratio calculation
        
        Args:
            client_gradients: 客户端梯度字典 {client_id: gradient}
            aggregated_gradient: 全局聚合梯度
            client_num_samples: 客户端样本数 (可选使用)
            
        Returns:
            contributions: 贡献度字典 {client_id: contribution}
        """
        contributions = {}
        
        for client_id, gradient in client_gradients.items():
            contribution = self.calculate_contribution(gradient, aggregated_gradient)
            contributions[client_id] = contribution
        
        return contributions
    
    def normalize_contributions(self, 
                               contributions: Dict[int, float]) -> Dict[int, float]:
        """
        归一化贡献度到 [0, 1] (Min-Max归一化)
        Normalize contributions to [0, 1] using Min-Max normalization
        
        公式 / Formula:
            φ̂_i = (φ_i - φ_min) / (φ_max - φ_min + ε)
        
        这对应于方法论文档中的"全局归一化贡献分"
        This corresponds to "Global Normalized Contribution" in methodology document
        
        Args:
            contributions: 原始贡献度 {client_id: contribution}
            
        Returns:
            normalized: 归一化后的贡献度 {client_id: normalized_contribution}
        """
        if not contributions:
            return {}
        
        values = list(contributions.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val - min_val < self.epsilon:
            # 所有贡献度相同，返回中间值
            # All contributions are identical, return midpoint
            return {cid: 0.5 for cid in contributions.keys()}
        
        normalized = {}
        for client_id, contribution in contributions.items():
            normalized[client_id] = (contribution - min_val) / (max_val - min_val + self.epsilon)
        
        return normalized
    
    def get_contribution_statistics(self, 
                                   contributions: Dict[int, float]) -> Dict:
        """
        获取贡献度统计信息
        Get contribution statistics
        
        Args:
            contributions: 贡献度字典
            
        Returns:
            统计信息字典 / Statistics dictionary
        """
        if not contributions:
            return {}
        
        values = list(contributions.values())
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }