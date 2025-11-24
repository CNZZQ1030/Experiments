"""
incentive/sparsification_distributor.py
稀疏化差异模型分发器 / Sparsification-based Differentiated Model Distributor

实现基于贡献度的模型稀疏化策略 / Implements contribution-based model sparsification strategy
高贡献客户端获得更完整的模型（低稀疏率）/ High contributors get more complete models (low sparsity)
低贡献客户端获得更稀疏的模型（高稀疏率）/ Low contributors get sparser models (high sparsity)
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class SparsificationDistributor:
    """
    稀疏化模型分发器 / Sparsification Model Distributor
    
    核心思想 / Core Idea:
    - 根据客户端贡献度计算保留率 / Calculate keep ratio based on contribution
    - 对全局模型进行稀疏化处理 / Apply sparsification to global model
    - 高贡献客户端获得接近完整的模型 / High contributors get nearly complete model
    - 低贡献客户端获得高度稀疏的模型 / Low contributors get highly sparse model
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 device: torch.device,
                 sparsity_ranges: Dict[str, Tuple[float, float]] = None,
                 min_keep_ratio: float = 0.1,
                 lambda_coefficient: float = 2.0,
                 sparsification_mode: str = "magnitude",
                 layer_importance_weights: Dict[str, float] = None):
        """
        初始化稀疏化分发器 / Initialize sparsification distributor
        
        Args:
            base_model: 基础模型 / Base model
            device: 计算设备 / Computing device
            sparsity_ranges: 各等级的稀疏率范围 / Sparsity ranges for each level
            min_keep_ratio: 最低保留率 / Minimum keep ratio
            lambda_coefficient: 调节系数λ / Adjustment coefficient λ
            sparsification_mode: 稀疏化模式 / Sparsification mode
            layer_importance_weights: 层重要性权重 / Layer importance weights
        """
        self.base_model = copy.deepcopy(base_model).to(device)
        self.device = device
        self.min_keep_ratio = min_keep_ratio
        self.lambda_coefficient = lambda_coefficient
        self.sparsification_mode = sparsification_mode
        
        # 默认稀疏率范围 / Default sparsity ranges
        if sparsity_ranges is None:
            self.sparsity_ranges = {
                'diamond': (0.0, 0.1),
                'gold': (0.1, 0.3),
                'silver': (0.3, 0.6),
                'bronze': (0.6, 0.95)
            }
        else:
            self.sparsity_ranges = sparsity_ranges
        
        # 层重要性权重 / Layer importance weights
        if layer_importance_weights is None:
            self.layer_importance_weights = {
                'conv': 1.0,
                'fc': 0.8,
                'bn': 0.0,  # BN层不稀疏化 / Don't sparsify BN layers
                'first': 0.5,
                'last': 0.5
            }
        else:
            self.layer_importance_weights = layer_importance_weights
        
        # 统计信息 / Statistics
        self.sparsification_stats = {}
        
        print(f"SparsificationDistributor initialized:")
        print(f"  Mode: {sparsification_mode}")
        print(f"  Min Keep Ratio: {min_keep_ratio}")
        print(f"  Lambda: {lambda_coefficient}")
        print(f"  Sparsity Ranges: {self.sparsity_ranges}")
    
    def calculate_keep_ratio(self, 
                           relative_contribution: float,
                           membership_level: str) -> float:
        """
        计算模型保留率 / Calculate model keep ratio
        
        公式 / Formula: α_i = Min_Keep + (1 - Min_Keep) × (r_i)^λ
        
        Args:
            relative_contribution: 相对贡献度 r_i ∈ [0,1] / Relative contribution
            membership_level: 会员等级 / Membership level
            
        Returns:
            保留率 α_i / Keep ratio
        """
        # 基于公式计算基础保留率 / Calculate base keep ratio using formula
        base_keep_ratio = self.min_keep_ratio + \
                         (1 - self.min_keep_ratio) * \
                         (relative_contribution ** self.lambda_coefficient)
        
        # 根据会员等级调整范围 / Adjust range based on membership level
        min_sparse, max_sparse = self.sparsity_ranges.get(membership_level, (0.6, 0.95))
        
        # 转换稀疏率到保留率 / Convert sparsity rate to keep ratio
        max_keep = 1.0 - min_sparse
        min_keep = 1.0 - max_sparse
        
        # 在等级范围内进行线性插值 / Linear interpolation within level range
        # 确保保留率在该等级的合理范围内 / Ensure keep ratio is within reasonable range for the level
        keep_ratio = min_keep + (max_keep - min_keep) * relative_contribution
        
        # 结合基础保留率和等级范围 / Combine base ratio and level range
        final_keep_ratio = 0.5 * base_keep_ratio + 0.5 * keep_ratio
        
        # 确保在有效范围内 / Ensure within valid range
        final_keep_ratio = max(self.min_keep_ratio, min(1.0, final_keep_ratio))
        
        return final_keep_ratio
    
    def magnitude_based_sparsification(self,
                                      weights: torch.Tensor,
                                      keep_ratio: float) -> torch.Tensor:
        """
        基于权重大小的稀疏化 / Magnitude-based sparsification
        保留最大的keep_ratio比例的权重 / Keep the largest keep_ratio proportion of weights
        
        Args:
            weights: 原始权重 / Original weights
            keep_ratio: 保留率 / Keep ratio
            
        Returns:
            稀疏化后的权重 / Sparsified weights
        """
        if keep_ratio >= 1.0:
            return weights
        
        # 展平权重 / Flatten weights
        weights_flat = weights.flatten()
        num_params = weights_flat.numel()
        
        # 计算需要保留的参数数量 / Calculate number of parameters to keep
        num_keep = max(1, int(num_params * keep_ratio))
        
        # 获取权重绝对值 / Get absolute values
        abs_weights = torch.abs(weights_flat)
        
        # 找到阈值 / Find threshold
        threshold = torch.topk(abs_weights, num_keep, largest=True, sorted=False)[0].min()
        
        # 创建掩码 / Create mask
        mask = torch.abs(weights) >= threshold
        
        # 应用掩码 / Apply mask
        sparse_weights = weights * mask.float()
        
        return sparse_weights
    
    def random_sparsification(self,
                            weights: torch.Tensor,
                            keep_ratio: float) -> torch.Tensor:
        """
        随机稀疏化 / Random sparsification
        随机保留keep_ratio比例的权重 / Randomly keep keep_ratio proportion of weights
        
        Args:
            weights: 原始权重 / Original weights
            keep_ratio: 保留率 / Keep ratio
            
        Returns:
            稀疏化后的权重 / Sparsified weights
        """
        if keep_ratio >= 1.0:
            return weights
        
        # 创建随机掩码 / Create random mask
        mask = torch.rand_like(weights) < keep_ratio
        
        # 应用掩码 / Apply mask
        sparse_weights = weights * mask.float()
        
        return sparse_weights
    
    def structured_sparsification(self,
                                weights: torch.Tensor,
                                keep_ratio: float,
                                dim: int = 0) -> torch.Tensor:
        """
        结构化稀疏化 / Structured sparsification
        按维度（通道/滤波器）进行稀疏化 / Sparsify by dimension (channel/filter)
        
        Args:
            weights: 原始权重 / Original weights
            keep_ratio: 保留率 / Keep ratio
            dim: 稀疏化维度 / Dimension to sparsify
            
        Returns:
            稀疏化后的权重 / Sparsified weights
        """
        if keep_ratio >= 1.0:
            return weights
        
        # 对于卷积层权重 [out_channels, in_channels, H, W]
        # For conv weights [out_channels, in_channels, H, W]
        if len(weights.shape) == 4:
            # 计算每个滤波器的L2范数 / Calculate L2 norm for each filter
            norms = torch.norm(weights.view(weights.shape[0], -1), dim=1)
            num_filters = weights.shape[0]
            num_keep = max(1, int(num_filters * keep_ratio))
            
            # 找到要保留的滤波器 / Find filters to keep
            _, keep_indices = torch.topk(norms, num_keep, largest=True)
            
            # 创建掩码 / Create mask
            mask = torch.zeros_like(weights)
            mask[keep_indices] = 1.0
            
            return weights * mask
        
        # 对于全连接层权重 [out_features, in_features]
        # For FC weights [out_features, in_features]
        elif len(weights.shape) == 2:
            # 计算每行的L2范数 / Calculate L2 norm for each row
            norms = torch.norm(weights, dim=1)
            num_rows = weights.shape[0]
            num_keep = max(1, int(num_rows * keep_ratio))
            
            # 找到要保留的行 / Find rows to keep
            _, keep_indices = torch.topk(norms, num_keep, largest=True)
            
            # 创建掩码 / Create mask
            mask = torch.zeros_like(weights)
            mask[keep_indices, :] = 1.0
            
            return weights * mask
        
        # 默认使用magnitude方法 / Default to magnitude method
        return self.magnitude_based_sparsification(weights, keep_ratio)
    
    def apply_sparsification_to_model(self,
                                     model_weights: Dict[str, torch.Tensor],
                                     keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        对整个模型应用稀疏化 / Apply sparsification to entire model
        
        Args:
            model_weights: 模型权重字典 / Model weights dictionary
            keep_ratio: 保留率 / Keep ratio
            
        Returns:
            稀疏化后的模型权重 / Sparsified model weights
        """
        sparse_weights = {}
        total_params = 0
        kept_params = 0
        
        for name, weight in model_weights.items():
            # 跳过批归一化层和偏置 / Skip batch normalization and bias
            if 'bn' in name.lower() or 'bias' in name or 'running' in name or 'num_batches' in name:
                sparse_weights[name] = weight.clone()
                continue
            
            # 根据层名称调整保留率 / Adjust keep ratio based on layer name
            layer_keep_ratio = keep_ratio
            
            # 第一层和最后一层特殊处理 / Special handling for first and last layers
            layer_names = list(model_weights.keys())
            weight_layers = [n for n in layer_names if 'weight' in n and 'bn' not in n.lower()]
            
            if weight_layers and name == weight_layers[0]:
                # 第一层：减少稀疏化 / First layer: reduce sparsification
                layer_keep_ratio = keep_ratio + (1 - keep_ratio) * 0.5
            elif weight_layers and name == weight_layers[-1]:
                # 最后一层：减少稀疏化 / Last layer: reduce sparsification
                layer_keep_ratio = keep_ratio + (1 - keep_ratio) * 0.5
            
            # 应用稀疏化 / Apply sparsification
            if self.sparsification_mode == "magnitude":
                sparse_weight = self.magnitude_based_sparsification(weight, layer_keep_ratio)
            elif self.sparsification_mode == "random":
                sparse_weight = self.random_sparsification(weight, layer_keep_ratio)
            elif self.sparsification_mode == "structured":
                sparse_weight = self.structured_sparsification(weight, layer_keep_ratio)
            else:
                sparse_weight = weight.clone()
            
            sparse_weights[name] = sparse_weight
            
            # 统计稀疏化情况 / Statistics
            total_params += weight.numel()
            kept_params += (sparse_weight != 0).sum().item()
        
        # 计算实际稀疏率 / Calculate actual sparsity
        actual_keep_ratio = kept_params / total_params if total_params > 0 else 1.0
        
        return sparse_weights, actual_keep_ratio
    
    def create_sparsified_model(self,
                              client_id: int,
                              membership_level: str,
                              relative_contribution: float,
                              global_model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        为客户端创建稀疏化模型 / Create sparsified model for client
        
        Args:
            client_id: 客户端ID / Client ID
            membership_level: 会员等级 / Membership level
            relative_contribution: 相对贡献度 [0,1] / Relative contribution
            global_model_weights: 全局模型权重 / Global model weights
            
        Returns:
            稀疏化后的模型权重 / Sparsified model weights
        """
        # 计算保留率 / Calculate keep ratio
        keep_ratio = self.calculate_keep_ratio(relative_contribution, membership_level)
        
        # 应用稀疏化 / Apply sparsification
        sparse_weights, actual_keep_ratio = self.apply_sparsification_to_model(
            global_model_weights, keep_ratio
        )
        
        # 记录统计信息 / Record statistics
        self.sparsification_stats[client_id] = {
            'membership_level': membership_level,
            'relative_contribution': relative_contribution,
            'target_keep_ratio': keep_ratio,
            'actual_keep_ratio': actual_keep_ratio,
            'sparsity_rate': 1.0 - actual_keep_ratio
        }
        
        return sparse_weights
    
    def distribute_all_sparsified_models(self,
                                        client_levels: Dict[int, str],
                                        client_contributions: Dict[int, float],
                                        global_model_weights: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        为所有客户端分发稀疏化模型 / Distribute sparsified models to all clients
        
        Args:
            client_levels: 客户端等级 {client_id: level}
            client_contributions: 客户端贡献度 {client_id: contribution}
            global_model_weights: 全局模型权重 / Global model weights
            
        Returns:
            稀疏化模型字典 {client_id: sparsified_weights}
        """
        sparsified_models = {}
        
        # 计算相对贡献度（归一化到[0,1]）/ Calculate relative contributions (normalized to [0,1])
        if client_contributions:
            min_contrib = min(client_contributions.values())
            max_contrib = max(client_contributions.values())
            contrib_range = max_contrib - min_contrib
            
            if contrib_range > 0:
                relative_contributions = {
                    cid: (contrib - min_contrib) / contrib_range
                    for cid, contrib in client_contributions.items()
                }
            else:
                # 所有贡献度相同 / All contributions are the same
                relative_contributions = {cid: 0.5 for cid in client_contributions.keys()}
        else:
            relative_contributions = {}
        
        # 为每个客户端创建稀疏化模型 / Create sparsified model for each client
        for client_id in client_levels.keys():
            level = client_levels.get(client_id, 'bronze')
            relative_contrib = relative_contributions.get(client_id, 0.0)
            
            sparsified_model = self.create_sparsified_model(
                client_id=client_id,
                membership_level=level,
                relative_contribution=relative_contrib,
                global_model_weights=global_model_weights
            )
            
            sparsified_models[client_id] = sparsified_model
        
        return sparsified_models
    
    def get_sparsification_statistics(self) -> Dict:
        """
        获取稀疏化统计信息 / Get sparsification statistics
        
        Returns:
            统计信息 / Statistics
        """
        if not self.sparsification_stats:
            return {}
        
        stats_by_level = {'diamond': [], 'gold': [], 'silver': [], 'bronze': []}
        
        for client_stats in self.sparsification_stats.values():
            level = client_stats['membership_level']
            if level in stats_by_level:
                stats_by_level[level].append(client_stats)
        
        summary = {}
        for level, level_stats in stats_by_level.items():
            if level_stats:
                keep_ratios = [s['actual_keep_ratio'] for s in level_stats]
                sparsity_rates = [s['sparsity_rate'] for s in level_stats]
                summary[level] = {
                    'count': len(level_stats),
                    'avg_keep_ratio': np.mean(keep_ratios),
                    'std_keep_ratio': np.std(keep_ratios),
                    'avg_sparsity_rate': np.mean(sparsity_rates),
                    'std_sparsity_rate': np.std(sparsity_rates),
                    'expected_range': self.sparsity_ranges[level]
                }
        
        return {
            'by_level': summary,
            'total_clients': len(self.sparsification_stats),
            'mode': self.sparsification_mode,
            'lambda': self.lambda_coefficient,
            'min_keep_ratio': self.min_keep_ratio
        }
    
    def print_sparsification_summary(self):
        """打印稀疏化摘要 / Print sparsification summary"""
        stats = self.get_sparsification_statistics()
        
        if not stats:
            print("No sparsification statistics available.")
            return
        
        print(f"\n{'='*70}")
        print(f"Sparsification Summary")
        print(f"{'='*70}")
        print(f"Mode: {stats['mode']}")
        print(f"Lambda (λ): {stats['lambda']}")
        print(f"Min Keep Ratio: {stats['min_keep_ratio']}")
        print(f"Total Clients: {stats['total_clients']}")
        
        print(f"\nStatistics by Membership Level:")
        print(f"{'Level':<10} {'Count':<8} {'Avg Keep':<10} {'Avg Sparse':<12} {'Expected Range'}")
        print(f"{'-'*70}")
        
        for level in ['diamond', 'gold', 'silver', 'bronze']:
            if level in stats['by_level']:
                level_stats = stats['by_level'][level]
                expected_min, expected_max = level_stats['expected_range']
                print(f"{level.capitalize():<10} {level_stats['count']:<8} "
                      f"{level_stats['avg_keep_ratio']:.3f}±{level_stats['std_keep_ratio']:.3f}  "
                      f"{level_stats['avg_sparsity_rate']:.3f}±{level_stats['std_sparsity_rate']:.3f}  "
                      f"[{expected_min:.2f}-{expected_max:.2f}]")
        
        print(f"{'='*70}")