"""
incentive/points_calculator.py (Updated for PDF Specification)
CGSV贡献度计算器 - 按照PDF公式实现 / CGSV Contribution Calculator - Per PDF Formula

公式 / Formula: s_{i,t} = max(0, CosineSimilarity(g_{i,t}, g_{agg,t}))
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple
from datetime import datetime


class CGSVContributionCalculator:
    """
    CGSV贡献度计算器 / CGSV Contribution Calculator
    
    按照PDF Definition 1实现瞬时贡献计算
    Implements instantaneous contribution calculation per PDF Definition 1
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        初始化CGSV计算器 / Initialize CGSV calculator
        
        Args:
            epsilon: 防止除零的小常数 / Small constant to prevent division by zero
        """
        self.epsilon = epsilon
        
        # 贡献历史记录 / Contribution history
        self.contribution_history = {}  # {client_id: [records]}
        
        # 每轮的统计信息 / Statistics per round
        self.round_statistics = []
        
        print("CGSVContributionCalculator initialized:")
        print("  Formula: s_{i,t} = max(0, CosineSimilarity(g_i, g_agg))")
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
        计算瞬时贡献分数 / Calculate instantaneous contribution score
        
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
        
        contribution = max(0.0, cos_sim)

        return contribution
    
    def calculate_all_contributions(self, 
                                   round_num: int,
                                   client_gradients: Dict[int, Dict[str, torch.Tensor]],
                                   aggregated_gradient: Dict[str, torch.Tensor]) -> Dict[int, float]:
        """
        计算所有客户端的瞬时贡献 / Calculate instantaneous contributions for all clients
        
        Args:
            round_num: 当前轮次 / Current round
            client_gradients: 所有客户端梯度 {client_id: gradient}
            aggregated_gradient: 聚合梯度 / Aggregated gradient
            
        Returns:
            贡献度字典 {client_id: s_{i,t}}
        """
        contributions = {}
        
        for client_id, client_gradient in client_gradients.items():
            contribution = self.calculate_instantaneous_contribution(
                client_gradient, aggregated_gradient
            )
            contributions[client_id] = contribution
            
            # 记录历史 / Record history
            if client_id not in self.contribution_history:
                self.contribution_history[client_id] = []
            
            self.contribution_history[client_id].append({
                'round': round_num,
                'contribution': contribution,
                'timestamp': datetime.now()
            })
        
        # 记录本轮统计 / Record round statistics
        if contributions:
            values = list(contributions.values())
            round_stats = {
                'round': round_num,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'num_clients': len(values)
            }
            self.round_statistics.append(round_stats)
            
            # 打印统计（每10轮）/ Print stats every 10 rounds
            if round_num == 1 or round_num % 10 == 0:
                print(f"\nRound {round_num} CGSV Statistics:")
                print(f"  Mean: {round_stats['mean']:.4f}, Std: {round_stats['std']:.4f}")
                print(f"  Range: [{round_stats['min']:.4f}, {round_stats['max']:.4f}]")
        
        return contributions
    
    def get_client_contribution_history(self, client_id: int) -> List[Dict]:
        """获取客户端贡献历史 / Get client contribution history"""
        return self.contribution_history.get(client_id, [])
    
    def get_contribution_statistics(self) -> Dict:
        """获取贡献度统计信息 / Get contribution statistics"""
        all_contributions = []
        
        for client_history in self.contribution_history.values():
            for record in client_history:
                all_contributions.append(record['contribution'])
        
        if not all_contributions:
            return {}
        
        return {
            'mean': np.mean(all_contributions),
            'std': np.std(all_contributions),
            'min': np.min(all_contributions),
            'max': np.max(all_contributions),
            'total_evaluations': len(all_contributions),
            'num_clients': len(self.contribution_history),
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
        print(f"\nOverall Statistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"{'='*70}")