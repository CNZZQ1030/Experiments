"""
incentive/points_calculator.py (重构版 - 论文CGSV实现)
CGSV贡献度计算器 - 按照论文公式实现并添加归一化
CGSV Contribution Calculator - Per Paper Formula with Normalization

核心改进:
1. 添加Min-Max归一化解决贡献度过于接近的问题
2. 支持多种归一化方法
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class CGSVContributionCalculator:
    """
    CGSV贡献度计算器 / CGSV Contribution Calculator
    
    按照论文Definition 1实现瞬时贡献计算，并添加归一化处理
    Implements instantaneous contribution calculation per Paper Definition 1 with normalization
    
    公式 / Formula: 
    - 原始: s_{i,t} = max(0, CosineSimilarity(g_{i,t}, g_{agg,t}))
    - 归一化后: s'_{i,t} = (s_{i,t} - min) / (max - min)
    """
    
    def __init__(self, epsilon: float = 1e-10, 
                 normalization_method: str = "softmax"):
        """
        初始化CGSV计算器 / Initialize CGSV calculator
        
        Args:
            epsilon: 防止除零的小常数 / Small constant to prevent division by zero
            normalization_method: 归一化方法 / Normalization method
                - "minmax": Min-Max归一化到[0,1] / Min-Max normalization to [0,1]
                - "zscore": Z-Score标准化 / Z-Score standardization
                - "softmax": Softmax归一化 / Softmax normalization
                - "none": 不归一化 / No normalization
        """
        self.epsilon = epsilon
        self.normalization_method = normalization_method
        
        # 贡献历史记录 / Contribution history
        self.contribution_history = {}  # {client_id: [records]}
        self.raw_contribution_history = {}  # 原始未归一化的贡献
        
        # 每轮的统计信息 / Statistics per round
        self.round_statistics = []
        
        print("CGSVContributionCalculator initialized:")
        print("  Formula: s_{i,t} = max(0, CosineSimilarity(g_i, g_agg))")
        print(f"  Normalization Method: {normalization_method}")
        print(f"  Epsilon: {epsilon}")
    
    def flatten_gradient(self, gradient: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将梯度字典展平为一维向量 / Flatten gradient dict to 1D vector
        
        Args:
            gradient: 梯度字典 / Gradient dictionary
            
        Returns:
            展平的梯度向量 / Flattened gradient vector
        """
        tensors = []
        for name in sorted(gradient.keys()):
            tensors.append(gradient[name].data.view(-1).float())
        return torch.cat(tensors)
    
    def calculate_cosine_similarity(self, gi: torch.Tensor, gagg: torch.Tensor) -> float:
        """
        计算余弦相似度 / Calculate cosine similarity
        
        公式: cos(g_i, g_agg) = <g_i, g_agg> / (||g_i|| * ||g_agg||)
        
        Args:
            gi: 客户端梯度 / Client gradient
            gagg: 聚合梯度 / Aggregated gradient
            
        Returns:
            余弦相似度值 [-1, 1] / Cosine similarity value [-1, 1]
        """
        cos_sim = F.cosine_similarity(gi.unsqueeze(0), gagg.unsqueeze(0), dim=1, eps=self.epsilon)
        return cos_sim.item()
    
    def calculate_instantaneous_contribution(self, 
                                            client_gradient: Dict[str, torch.Tensor],
                                            aggregated_gradient: Dict[str, torch.Tensor]) -> float:
        """
        计算瞬时贡献分数（原始值）/ Calculate instantaneous contribution score (raw)
        
        公式 / Formula: s_{i,t} = max(0, CosineSimilarity(g_{i,t}, g_{agg,t}))
        
        Args:
            client_gradient: 客户端梯度 / Client gradient
            aggregated_gradient: 聚合梯度 / Aggregated gradient
            
        Returns:
            瞬时贡献分数 [0, 1] / Instantaneous contribution score [0, 1]
        """
        gi_flat = self.flatten_gradient(client_gradient)
        gagg_flat = self.flatten_gradient(aggregated_gradient)
        
        cos_sim = self.calculate_cosine_similarity(gi_flat, gagg_flat)
        
        # 取max(0, cos_sim)确保非负
        contribution = max(0.0, cos_sim)

        return contribution
    
    def normalize_contributions_minmax(self, 
                                       raw_contributions: Dict[int, float]) -> Dict[int, float]:
        """
        Min-Max归一化贡献度 / Min-Max normalize contributions
        
        将贡献度强制拉伸到[0, 1]区间，最大化差异
        Force contributions to [0, 1] range, maximizing differentiation
        
        Args:
            raw_contributions: 原始贡献度 / Raw contributions
            
        Returns:
            归一化后的贡献度 / Normalized contributions
        """
        if not raw_contributions:
            return {}
        
        values = list(raw_contributions.values())
        min_val = np.min(values)
        max_val = np.max(values)
        
        # 避免除以零 / Avoid division by zero
        if max_val - min_val > 1e-9:
            normalized = {
                k: (v - min_val) / (max_val - min_val)
                for k, v in raw_contributions.items()
            }
        else:
            # 所有值相同，设为均匀分布 / All values same, set to uniform
            n = len(raw_contributions)
            normalized = {k: 1.0 / n for k in raw_contributions.keys()}
        
        return normalized
    
    def normalize_contributions_zscore(self, 
                                       raw_contributions: Dict[int, float]) -> Dict[int, float]:
        """
        Z-Score标准化贡献度 / Z-Score standardize contributions
        
        Args:
            raw_contributions: 原始贡献度 / Raw contributions
            
        Returns:
            标准化后的贡献度 / Standardized contributions
        """
        if not raw_contributions:
            return {}
        
        values = np.array(list(raw_contributions.values()))
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 1e-9:
            # Z-Score标准化后映射到[0,1]
            z_scores = (values - mean_val) / std_val
            # 使用sigmoid将z-score映射到[0,1]
            normalized_values = 1 / (1 + np.exp(-z_scores))
            normalized = {
                k: normalized_values[i]
                for i, k in enumerate(raw_contributions.keys())
            }
        else:
            n = len(raw_contributions)
            normalized = {k: 1.0 / n for k in raw_contributions.keys()}
        
        return normalized
    
    def normalize_contributions_softmax(self, 
                                        raw_contributions: Dict[int, float],
                                        temperature: float = 0.1) -> Dict[int, float]:
        """
        Softmax归一化贡献度 / Softmax normalize contributions
        
        使用较低的温度参数放大差异 / Use low temperature to amplify differences
        
        Args:
            raw_contributions: 原始贡献度 / Raw contributions
            temperature: 温度参数，越小差异越大 / Temperature, smaller = more difference
            
        Returns:
            归一化后的贡献度 / Normalized contributions
        """
        if not raw_contributions:
            return {}
        
        values = np.array(list(raw_contributions.values()))
        
        # Softmax with temperature
        exp_values = np.exp((values - np.max(values)) / temperature)
        softmax_values = exp_values / np.sum(exp_values)
        
        # 缩放到[0, 1]范围 / Scale to [0, 1] range
        min_sm = np.min(softmax_values)
        max_sm = np.max(softmax_values)
        
        if max_sm - min_sm > 1e-9:
            scaled_values = (softmax_values - min_sm) / (max_sm - min_sm)
        else:
            scaled_values = softmax_values
        
        normalized = {
            k: scaled_values[i]
            for i, k in enumerate(raw_contributions.keys())
        }
        
        return normalized
    
    def normalize_contributions(self, raw_contributions: Dict[int, float]) -> Dict[int, float]:
        """
        根据配置的方法归一化贡献度 / Normalize contributions based on configured method
        
        Args:
            raw_contributions: 原始贡献度 / Raw contributions
            
        Returns:
            归一化后的贡献度 / Normalized contributions
        """
        if self.normalization_method == "minmax":
            return self.normalize_contributions_minmax(raw_contributions)
        elif self.normalization_method == "zscore":
            return self.normalize_contributions_zscore(raw_contributions)
        elif self.normalization_method == "softmax":
            return self.normalize_contributions_softmax(raw_contributions)
        else:  # "none"
            return raw_contributions
    
    def calculate_all_contributions(self, 
                                   round_num: int,
                                   client_gradients: Dict[int, Dict[str, torch.Tensor]],
                                   aggregated_gradient: Dict[str, torch.Tensor]) -> Dict[int, float]:
        """
        计算所有客户端的瞬时贡献并归一化 / Calculate and normalize instantaneous contributions for all clients
        
        Args:
            round_num: 当前轮次 / Current round
            client_gradients: 所有客户端梯度 {client_id: gradient}
            aggregated_gradient: 聚合梯度 / Aggregated gradient
            
        Returns:
            归一化后的贡献度字典 {client_id: normalized_s_{i,t}}
        """
        # 第一步：计算原始贡献度 / Step 1: Calculate raw contributions
        raw_contributions = {}
        
        for client_id, client_gradient in client_gradients.items():
            contribution = self.calculate_instantaneous_contribution(
                client_gradient, aggregated_gradient
            )
            raw_contributions[client_id] = contribution
            
            # 记录原始历史 / Record raw history
            if client_id not in self.raw_contribution_history:
                self.raw_contribution_history[client_id] = []
            
            self.raw_contribution_history[client_id].append({
                'round': round_num,
                'raw_contribution': contribution,
                'timestamp': datetime.now()
            })
        
        # 第二步：归一化贡献度 / Step 2: Normalize contributions
        normalized_contributions = self.normalize_contributions(raw_contributions)
        
        # 记录归一化后的历史 / Record normalized history
        for client_id, contribution in normalized_contributions.items():
            if client_id not in self.contribution_history:
                self.contribution_history[client_id] = []
            
            self.contribution_history[client_id].append({
                'round': round_num,
                'contribution': contribution,
                'raw_contribution': raw_contributions[client_id],
                'timestamp': datetime.now()
            })
        
        # 记录本轮统计 / Record round statistics
        if raw_contributions:
            raw_values = list(raw_contributions.values())
            norm_values = list(normalized_contributions.values())
            
            round_stats = {
                'round': round_num,
                'raw_mean': np.mean(raw_values),
                'raw_std': np.std(raw_values),
                'raw_min': np.min(raw_values),
                'raw_max': np.max(raw_values),
                'normalized_mean': np.mean(norm_values),
                'normalized_std': np.std(norm_values),
                'normalized_min': np.min(norm_values),
                'normalized_max': np.max(norm_values),
                'num_clients': len(raw_values)
            }
            self.round_statistics.append(round_stats)
            
            # 打印统计 / Print stats
            if round_num == 1 or round_num % 10 == 0:
                print(f"\nRound {round_num} CGSV Statistics:")
                print(f"  Raw - Mean: {round_stats['raw_mean']:.6f}, Std: {round_stats['raw_std']:.6f}")
                print(f"  Raw - Range: [{round_stats['raw_min']:.6f}, {round_stats['raw_max']:.6f}]")
                print(f"  Normalized ({self.normalization_method}) - Mean: {round_stats['normalized_mean']:.4f}, Std: {round_stats['normalized_std']:.4f}")
                print(f"  Normalized - Range: [{round_stats['normalized_min']:.4f}, {round_stats['normalized_max']:.4f}]")
        
        return normalized_contributions
    
    def get_client_contribution_history(self, client_id: int) -> List[Dict]:
        """获取客户端贡献历史 / Get client contribution history"""
        return self.contribution_history.get(client_id, [])
    
    def get_contribution_statistics(self) -> Dict:
        """获取贡献度统计信息 / Get contribution statistics"""
        all_contributions = []
        all_raw_contributions = []
        
        for client_history in self.contribution_history.values():
            for record in client_history:
                all_contributions.append(record['contribution'])
                all_raw_contributions.append(record['raw_contribution'])
        
        if not all_contributions:
            return {}
        
        return {
            'normalized_mean': np.mean(all_contributions),
            'normalized_std': np.std(all_contributions),
            'normalized_min': np.min(all_contributions),
            'normalized_max': np.max(all_contributions),
            'raw_mean': np.mean(all_raw_contributions),
            'raw_std': np.std(all_raw_contributions),
            'raw_min': np.min(all_raw_contributions),
            'raw_max': np.max(all_raw_contributions),
            'total_evaluations': len(all_contributions),
            'num_clients': len(self.contribution_history),
            'normalization_method': self.normalization_method,
            'round_statistics': self.round_statistics
        }
    
    def print_summary(self):
        """打印贡献度统计摘要 / Print contribution statistics summary"""
        stats = self.get_contribution_statistics()
        
        if not stats:
            print("No contribution data available.")
            return
        
        print(f"\n{'='*70}")
        print(f"CGSV Contribution Summary")
        print(f"{'='*70}")
        print(f"Total Evaluations: {stats['total_evaluations']}")
        print(f"Number of Clients: {stats['num_clients']}")
        print(f"Normalization Method: {stats['normalization_method']}")
        print(f"\nRaw Statistics (before normalization):")
        print(f"  Mean: {stats['raw_mean']:.6f}")
        print(f"  Std:  {stats['raw_std']:.6f}")
        print(f"  Range: [{stats['raw_min']:.6f}, {stats['raw_max']:.6f}]")
        print(f"\nNormalized Statistics:")
        print(f"  Mean: {stats['normalized_mean']:.4f}")
        print(f"  Std:  {stats['normalized_std']:.4f}")
        print(f"  Range: [{stats['normalized_min']:.4f}, {stats['normalized_max']:.4f}]")
        print(f"{'='*70}")