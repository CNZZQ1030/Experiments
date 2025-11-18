"""
incentive/points_calculator.py (Updated with Relative Contribution Normalization)
CGSV贡献度计算器 - 增强版，解决贡献度过高问题
CGSV Contribution Calculator - Enhanced version, solving high contribution issue
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple
from datetime import datetime


class CGSVContributionCalculator:
    """
    CGSV贡献度计算器 - 相对归一化版本
    CGSV Contribution Calculator - Relative Normalization Version
    
    核心改进 / Core Improvements:
    1. 计算原始CGSV (余弦相似度) / Calculate raw CGSV (cosine similarity)
    2. 相对归一化：contribution_relative = contribution / mean_contribution
       Relative normalization: contribution_relative = contribution / mean_contribution
    3. 突出贡献度差异 / Highlight contribution differences
    """
    
    def __init__(self, epsilon: float = 1e-10, 
                 use_relative_normalization: bool = True,
                 normalization_method: str = 'mean'):
        """
        初始化CGSV计算器 / Initialize CGSV calculator
        
        Args:
            epsilon: 防止除零的小常数 / Small constant to prevent division by zero
            use_relative_normalization: 是否使用相对归一化 / Whether to use relative normalization
            normalization_method: 归一化方法 / Normalization method
                - 'mean': 除以平均值 / Divide by mean
                - 'minmax': 最小-最大归一化 / Min-max normalization
                - 'zscore': Z-score标准化 / Z-score standardization
        """
        self.epsilon = epsilon
        self.use_relative_normalization = use_relative_normalization
        self.normalization_method = normalization_method
        
        # 贡献历史记录 / Contribution history
        self.contribution_history = {}  # {client_id: [records]}
        
        # 每轮的统计信息 / Statistics per round
        self.round_statistics = []
        
        print(f"CGSVContributionCalculator initialized:")
        print(f"  Relative Normalization: {use_relative_normalization}")
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
        return torch.cat([gradient[name].data.view(-1) for name in gradient])
    
    def calculate_raw_cosine_similarity(self, gi: torch.Tensor, gagg: torch.Tensor) -> float:
        """
        计算原始余弦相似度 / Calculate raw cosine similarity
        
        公式: cos(gi, gagg) = <gi, gagg> / (||gi|| * ||gagg||)
        
        Args:
            gi: 客户端梯度 / Client gradient
            gagg: 聚合梯度 / Aggregated gradient
            
        Returns:
            余弦相似度值 [-1, 1] / Cosine similarity value [-1, 1]
        """
        cos_sim = F.cosine_similarity(gi, gagg, 0, self.epsilon)
        return cos_sim.item()
    
    def normalize_contributions(self, raw_contributions: List[float]) -> List[float]:
        """
        对贡献度进行归一化处理 / Normalize contributions
        
        Args:
            raw_contributions: 原始贡献度列表 / Raw contribution list
            
        Returns:
            归一化后的贡献度 / Normalized contributions
        """
        raw_array = np.array(raw_contributions)
        
        if self.normalization_method == 'mean':
            # 相对于平均值的归一化 / Normalization relative to mean
            mean_val = np.mean(raw_array)
            if mean_val > self.epsilon:
                normalized = raw_array / mean_val
            else:
                normalized = raw_array
                
        elif self.normalization_method == 'minmax':
            # 最小-最大归一化到[0, 1] / Min-max normalization to [0, 1]
            min_val = np.min(raw_array)
            max_val = np.max(raw_array)
            if max_val - min_val > self.epsilon:
                normalized = (raw_array - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(raw_array) * 0.5
                
        elif self.normalization_method == 'zscore':
            # Z-score标准化 / Z-score standardization
            mean_val = np.mean(raw_array)
            std_val = np.std(raw_array)
            if std_val > self.epsilon:
                normalized = (raw_array - mean_val) / std_val
                # 转换到正值范围 / Convert to positive range
                normalized = normalized - np.min(normalized) + 0.1
            else:
                normalized = np.ones_like(raw_array)
        else:
            normalized = raw_array
        
        return normalized.tolist()
    
    def calculate_contributions_batch(self, round_num: int,
                                    client_gradients: Dict[int, Dict[str, torch.Tensor]],
                                    aggregated_gradient: Dict[str, torch.Tensor]) -> Dict[int, float]:
        """
        批量计算所有客户端的贡献度（新增方法）/ Calculate contributions for all clients in batch
        
        这个方法解决了需求4中的映射问题：
        1. 先计算所有原始CGSV
        2. 统一进行相对归一化
        3. 返回归一化后的贡献度
        
        Args:
            round_num: 当前轮次 / Current round
            client_gradients: 所有客户端梯度 {client_id: gradient}
            aggregated_gradient: 聚合梯度 / Aggregated gradient
            
        Returns:
            归一化后的贡献度 {client_id: normalized_contribution}
        """
        # 展平聚合梯度
        gagg_flat = self.flatten_gradient(aggregated_gradient)
        
        # 步骤1: 计算所有原始CGSV
        raw_contributions = {}
        client_ids = list(client_gradients.keys())
        
        for client_id, client_gradient in client_gradients.items():
            gi_flat = self.flatten_gradient(client_gradient)
            raw_cos_sim = self.calculate_raw_cosine_similarity(gi_flat, gagg_flat)
            # 转换到[0, 1]范围
            raw_contribution = max(0.0, (raw_cos_sim + 1.0) / 2.0)
            raw_contributions[client_id] = raw_contribution
        
        # 步骤2: 相对归一化（如果启用）
        if self.use_relative_normalization and len(raw_contributions) > 1:
            raw_values = [raw_contributions[cid] for cid in client_ids]
            normalized_values = self.normalize_contributions(raw_values)
            
            normalized_contributions = {
                client_ids[i]: normalized_values[i]
                for i in range(len(client_ids))
            }
        else:
            normalized_contributions = raw_contributions
        
        # 步骤3: 记录统计信息
        raw_values_list = list(raw_contributions.values())
        norm_values_list = list(normalized_contributions.values())
        
        round_stats = {
            'round': round_num,
            'raw_mean': np.mean(raw_values_list),
            'raw_std': np.std(raw_values_list),
            'raw_min': np.min(raw_values_list),
            'raw_max': np.max(raw_values_list),
            'normalized_mean': np.mean(norm_values_list),
            'normalized_std': np.std(norm_values_list),
            'normalized_min': np.min(norm_values_list),
            'normalized_max': np.max(norm_values_list),
        }
        self.round_statistics.append(round_stats)
        
        # 步骤4: 记录每个客户端的历史
        for client_id in client_ids:
            if client_id not in self.contribution_history:
                self.contribution_history[client_id] = []
            
            self.contribution_history[client_id].append({
                'round': round_num,
                'raw_contribution': raw_contributions[client_id],
                'normalized_contribution': normalized_contributions[client_id],
                'timestamp': datetime.now()
            })
        
        # 打印统计信息（每10轮或第1轮）
        if round_num == 1 or round_num % 10 == 0:
            print(f"\nRound {round_num} Contribution Statistics:")
            print(f"  Raw CGSV    - Mean: {round_stats['raw_mean']:.4f}, "
                  f"Std: {round_stats['raw_std']:.4f}, "
                  f"Range: [{round_stats['raw_min']:.4f}, {round_stats['raw_max']:.4f}]")
            if self.use_relative_normalization:
                print(f"  Normalized  - Mean: {round_stats['normalized_mean']:.4f}, "
                      f"Std: {round_stats['normalized_std']:.4f}, "
                      f"Range: [{round_stats['normalized_min']:.4f}, {round_stats['normalized_max']:.4f}]")
        
        return normalized_contributions
    
    def calculate_contribution(self, client_id: int, round_num: int,
                             client_gradient: Dict[str, torch.Tensor],
                             aggregated_gradient: Dict[str, torch.Tensor],
                             all_gradients: List[Dict[str, torch.Tensor]]) -> float:
        """
        计算单个客户端的CGSV贡献度（兼容性方法）
        Calculate single client's CGSV contribution (compatibility method)
        
        注意：建议使用calculate_contributions_batch以获得更好的归一化效果
        Note: Recommend using calculate_contributions_batch for better normalization
        """
        gi_flat = self.flatten_gradient(client_gradient)
        gagg_flat = self.flatten_gradient(aggregated_gradient)
        
        raw_cos_sim = self.calculate_raw_cosine_similarity(gi_flat, gagg_flat)
        contribution = max(0.0, (raw_cos_sim + 1.0) / 2.0)
        
        # 记录历史
        if client_id not in self.contribution_history:
            self.contribution_history[client_id] = []
        
        self.contribution_history[client_id].append({
            'round': round_num,
            'raw_contribution': contribution,
            'normalized_contribution': contribution,  # 单独计算时无法归一化
            'timestamp': datetime.now()
        })
        
        return contribution
    
    def get_client_contribution_history(self, client_id: int) -> List[Dict]:
        """获取客户端贡献历史 / Get client contribution history"""
        return self.contribution_history.get(client_id, [])
    
    def get_contribution_statistics(self) -> Dict:
        """获取贡献度统计信息 / Get contribution statistics"""
        all_raw = []
        all_normalized = []
        
        for client_history in self.contribution_history.values():
            for record in client_history:
                all_raw.append(record['raw_contribution'])
                all_normalized.append(record['normalized_contribution'])
        
        if not all_raw:
            return {}
        
        stats = {
            'raw_contributions': {
                'mean': np.mean(all_raw),
                'std': np.std(all_raw),
                'min': np.min(all_raw),
                'max': np.max(all_raw),
            },
            'normalized_contributions': {
                'mean': np.mean(all_normalized),
                'std': np.std(all_normalized),
                'min': np.min(all_normalized),
                'max': np.max(all_normalized),
            },
            'total_evaluations': len(all_raw),
            'num_clients': len(self.contribution_history),
            'round_statistics': self.round_statistics
        }
        
        return stats
    
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
        
        print(f"\nRaw CGSV Contributions:")
        print(f"  Mean: {stats['raw_contributions']['mean']:.4f}")
        print(f"  Std:  {stats['raw_contributions']['std']:.4f}")
        print(f"  Min:  {stats['raw_contributions']['min']:.4f}")
        print(f"  Max:  {stats['raw_contributions']['max']:.4f}")
        
        if self.use_relative_normalization:
            print(f"\nNormalized Contributions:")
            print(f"  Mean: {stats['normalized_contributions']['mean']:.4f}")
            print(f"  Std:  {stats['normalized_contributions']['std']:.4f}")
            print(f"  Min:  {stats['normalized_contributions']['min']:.4f}")
            print(f"  Max:  {stats['normalized_contributions']['max']:.4f}")
        
        print(f"{'='*70}")