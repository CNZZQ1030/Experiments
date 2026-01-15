"""
incentive/points_calculator.py
CGSV贡献度计算器 - 梯度版本
CGSV Contribution Calculator - Gradient Version
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
    
    计算客户端梯度与全局梯度的余弦相似度作为贡献度
    
    公式 / Formula:
        CGSV_i = cos(Δw_i, Δw_global) = (Δw_i · Δw_global) / (||Δw_i|| * ||Δw_global||)
    """
    
    def __init__(self):
        """初始化CGSV计算器"""
        self.epsilon = IncentiveConfig.CGSV_EPSILON
    
    def flatten_gradient(self, gradient: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将梯度字典展平为一维张量
        
        Args:
            gradient: 梯度字典
        
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
        
        Args:
            gradient_a: 梯度A
            gradient_b: 梯度B
        
        Returns:
            similarity: 余弦相似度 [-1, 1]
        """
        # 计算点积
        dot_product = torch.dot(gradient_a, gradient_b)
        
        # 计算范数
        norm_a = torch.norm(gradient_a)
        norm_b = torch.norm(gradient_b)
        
        # 避免除零
        if norm_a < self.epsilon or norm_b < self.epsilon:
            return 0.0
        
        # 余弦相似度
        similarity = dot_product / (norm_a * norm_b)
        
        return float(similarity.item())
    
    def calculate_contribution(self,
                              client_gradient: Dict[str, torch.Tensor],
                              aggregated_gradient: Dict[str, torch.Tensor]) -> float:
        """
        计算单个客户端的贡献度
        
        Args:
            client_gradient: 客户端梯度
            aggregated_gradient: 全局聚合梯度
        
        Returns:
            contribution: 贡献度值
        """
        # 展平梯度
        flat_client = self.flatten_gradient(client_gradient)
        flat_global = self.flatten_gradient(aggregated_gradient)
        
        # 计算余弦相似度
        similarity = self.cosine_similarity(flat_client, flat_global)
        
        # 归一化到 [0, 1]
        # cos ∈ [-1, 1] → contribution ∈ [0, 1]
        contribution = (similarity + 1.0) / 2.0
        
        return contribution
    
    def calculate_all_contributions(self,
                                   client_gradients: Dict[int, Dict[str, torch.Tensor]],
                                   aggregated_gradient: Dict[str, torch.Tensor],
                                   client_num_samples: Dict[int, int]) -> Dict[int, float]:
        """
        计算所有客户端的贡献度
        
        Args:
            client_gradients: 客户端梯度字典 {client_id: gradient}
            aggregated_gradient: 全局聚合梯度
            client_num_samples: 客户端样本数
        
        Returns:
            contributions: 贡献度字典 {client_id: contribution}
        """
        contributions = {}
        
        for client_id, gradient in client_gradients.items():
            contribution = self.calculate_contribution(gradient, aggregated_gradient)
            
            # 可选：根据样本数加权
            # num_samples = client_num_samples.get(client_id, 1)
            # weighted_contribution = contribution * np.sqrt(num_samples)
            
            contributions[client_id] = contribution
        
        # 归一化（可选）
        # contributions = self.normalize_contributions(contributions)
        
        return contributions
    
    def normalize_contributions(self, 
                               contributions: Dict[int, float]) -> Dict[int, float]:
        """
        归一化贡献度到 [0, 1]
        
        Args:
            contributions: 原始贡献度
        
        Returns:
            normalized: 归一化后的贡献度
        """
        if not contributions:
            return {}
        
        values = list(contributions.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val - min_val < self.epsilon:
            # 所有贡献度相同
            return {cid: 0.5 for cid in contributions.keys()}
        
        normalized = {}
        for client_id, contribution in contributions.items():
            normalized[client_id] = (contribution - min_val) / (max_val - min_val)
        
        return normalized