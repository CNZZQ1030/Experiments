"""
incentive/sparsification_distributor.py - 层级约束动态梯度奖励（重构版）
Tier-Constrained Dynamic Gradient Reward Distributor (Refactored)

基于NeurIPS 2021论文"Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"
Based on NeurIPS 2021 paper "Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"

核心创新 / Core Innovations:
1. 层级作为稀疏率的上下界（Bounds）而非固定值
   Tiers as bounds for keep ratios instead of fixed values
   
2. 组内插值（Intra-Tier Interpolation）实现连续映射
   Intra-tier interpolation for continuous mapping
   
3. 大幅降低低贡献客户端的参数保留率以提高PCC
   Significantly reduce keep ratio for low-contribution clients to improve PCC
   
4. 基于幅度的稀疏化（Magnitude-based Pruning）保留最重要的参数
   Magnitude-based pruning to retain most important parameters

公式说明 / Formula Description:
1. 全局归一化贡献分 / Global normalized contribution:
   φ̂_i = (φ_i - φ_min) / (φ_max - φ_min + ε)

2. 组内相对位置 / Intra-tier relative position:
   P_i = (φ_i - min(φ ∈ L)) / (max(φ ∈ L) - min(φ ∈ L) + ε)

3. 最终保留率 / Final keep ratio:
   s_i = S^L_low + P_i × (S^L_high - S^L_low)
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IncentiveConfig


class TierConstrainedGradientDistributor:
    """
    层级约束动态梯度奖励分发器
    Tier-Constrained Dynamic Gradient Reward Distributor
    
    核心流程 / Core Workflow:
    1. 接收所有客户端的CGSV贡献度分数
    2. 根据排名分配到不同等级（Gold/Silver/Bronze）
    3. 在等级内部进行线性插值计算精确的保留率
    4. 对全局聚合梯度进行差异化稀疏
    5. 分发稀疏化梯度给不同客户端
    
    Receive CGSV contribution scores -> Assign to tiers -> 
    Intra-tier interpolation -> Differential sparsification -> Distribute
    """
    
    def __init__(self, device: torch.device, 
                 tier_config: str = "default",
                 verbose: bool = True):
        """
        初始化层级约束梯度分发器
        Initialize Tier-Constrained Gradient Distributor
        
        Args:
            device: 计算设备 / Computing device
            tier_config: 层级配置选项 / Tier configuration option
                - "default": 默认配置 / Default configuration
                - "aggressive": 更激进的差异化 / More aggressive differentiation
                - "moderate": 更温和的差异化 / More moderate differentiation
            verbose: 是否打印详细信息 / Whether to print verbose info
        """
        self.device = device
        self.verbose = verbose
        
        # 选择层级配置 / Select tier configuration
        if tier_config == "aggressive":
            self.tier_keep_ratio_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES_AGGRESSIVE
        elif tier_config == "moderate":
            self.tier_keep_ratio_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES_MODERATE
        else:
            self.tier_keep_ratio_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES
        
        # 层级百分位配置 / Tier percentile configuration
        self.tier_percentiles = IncentiveConfig.LEVEL_PERCENTILES
        
        # 全局保留率限制 / Global keep ratio limits
        self.min_keep_ratio = IncentiveConfig.MIN_KEEP_RATIO
        self.max_keep_ratio = IncentiveConfig.MAX_KEEP_RATIO
        
        # 稀疏化模式 / Sparsification mode
        self.sparsification_mode = IncentiveConfig.SPARSIFICATION_MODE
        
        # 插值配置 / Interpolation configuration
        self.interpolation_method = IncentiveConfig.INTERPOLATION_METHOD
        self.interpolation_lambda = IncentiveConfig.INTERPOLATION_LAMBDA
        
        # epsilon防止除零 / Epsilon to prevent division by zero
        self.epsilon = IncentiveConfig.CGSV_EPSILON
        
        # 统计信息 / Statistics
        self.sparsification_stats = defaultdict(list)
        self.round_stats = {}
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Tier-Constrained Dynamic Gradient Distributor Initialized")
            print(f"层级约束动态梯度分发器已初始化")
            print(f"{'='*70}")
            print(f"Configuration / 配置:")
            print(f"  Tier Config / 层级配置: {tier_config}")
            print(f"  Sparsification Mode / 稀疏化模式: {self.sparsification_mode}")
            print(f"  Interpolation Method / 插值方法: {self.interpolation_method}")
            print(f"\nTier Keep Ratio Ranges / 层级保留率范围:")
            for tier, (low, high) in self.tier_keep_ratio_ranges.items():
                print(f"  {tier.capitalize():8s}: [{low:.2f}, {high:.2f}] "
                      f"(Sparsity: [{1-high:.2f}, {1-low:.2f}])")
            print(f"{'='*70}\n")
    
    def assign_clients_to_tiers(self, 
                                contributions: Dict[int, float]) -> Dict[str, List[Tuple[int, float]]]:
        """
        将客户端分配到不同层级
        Assign clients to different tiers based on contribution scores
        
        使用相对排名进行分配 / Use relative ranking for assignment
        
        Args:
            contributions: 客户端贡献度字典 {client_id: contribution_score}
            
        Returns:
            tier_groups: 层级分组 {tier_name: [(client_id, score), ...]}
        """
        if not contributions:
            return {'gold': [], 'silver': [], 'bronze': []}
        
        # 按贡献度降序排序 / Sort by contribution descending
        sorted_clients = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        total_clients = len(sorted_clients)
        
        # 初始化层级分组 / Initialize tier groups
        tier_groups = {'gold': [], 'silver': [], 'bronze': []}
        
        for rank, (client_id, score) in enumerate(sorted_clients):
            # 计算排名百分位 (0=最低, 1=最高)
            # Calculate ranking percentile (0=lowest, 1=highest)
            percentile = (total_clients - 1 - rank) / max(1, total_clients - 1)
            
            # 根据百分位分配层级 / Assign tier based on percentile
            if percentile >= self.tier_percentiles['gold']:
                tier_groups['gold'].append((client_id, score))
            elif percentile >= self.tier_percentiles['silver']:
                tier_groups['silver'].append((client_id, score))
            else:
                tier_groups['bronze'].append((client_id, score))
        
        return tier_groups
    
    def calculate_intra_tier_keep_ratio(self,
                                        client_score: float,
                                        tier_scores: List[float],
                                        tier_name: str) -> float:
        """
        计算组内插值的保留率
        Calculate keep ratio using intra-tier interpolation
        
        公式 / Formula:
        1. 计算组内相对位置 / Calculate intra-tier relative position:
           P_i = (φ_i - min(φ ∈ L)) / (max(φ ∈ L) - min(φ ∈ L) + ε)
        
        2. 线性插值计算保留率 / Linear interpolation for keep ratio:
           s_i = S^L_low + P_i × (S^L_high - S^L_low)
        
        Args:
            client_score: 客户端的贡献度分数 / Client's contribution score
            tier_scores: 该层级所有客户端的分数列表 / List of all scores in this tier
            tier_name: 层级名称 / Tier name
            
        Returns:
            keep_ratio: 计算得到的保留率 / Calculated keep ratio
        """
        # 获取该层级的保留率范围 / Get keep ratio range for this tier
        lower_bound, upper_bound = self.tier_keep_ratio_ranges.get(
            tier_name, (0.10, 0.50)  # 默认为bronze / Default to bronze
        )
        
        # 如果组内只有一个客户端，返回中间值 / If only one client in tier, return midpoint
        if len(tier_scores) <= 1:
            return (lower_bound + upper_bound) / 2
        
        # 计算组内最小和最大分数 / Calculate min and max scores in tier
        min_score = min(tier_scores)
        max_score = max(tier_scores)
        
        # 计算组内相对位置 P_i / Calculate intra-tier relative position P_i
        if max_score - min_score < self.epsilon:
            # 如果分数都相同，返回中间值 / If scores are identical, return midpoint
            relative_position = 0.5
        else:
            relative_position = (client_score - min_score) / (max_score - min_score + self.epsilon)
        
        # 根据插值方法计算最终保留率 / Calculate final keep ratio based on interpolation method
        if self.interpolation_method == "power":
            # 幂律插值 / Power-law interpolation
            relative_position = np.power(relative_position, self.interpolation_lambda)
        
        # 线性插值：s_i = S^L_low + P_i × (S^L_high - S^L_low)
        # Linear interpolation
        keep_ratio = lower_bound + relative_position * (upper_bound - lower_bound)
        
        # 确保在全局范围内 / Ensure within global limits
        keep_ratio = np.clip(keep_ratio, self.min_keep_ratio, self.max_keep_ratio)
        
        return float(keep_ratio)
    
    def calculate_all_keep_ratios(self,
                                  contributions: Dict[int, float],
                                  membership_levels: Dict[int, str]) -> Dict[int, float]:
        """
        计算所有客户端的保留率
        Calculate keep ratios for all clients
        
        这是核心方法，实现层级约束动态梯度奖励
        This is the core method implementing Tier-Constrained Dynamic Gradient Reward
        
        Args:
            contributions: 客户端贡献度 {client_id: score}
            membership_levels: 客户端等级 {client_id: tier_name}
            
        Returns:
            keep_ratios: 保留率字典 {client_id: keep_ratio}
        """
        if not contributions:
            return {}
        
        # 步骤1: 将客户端分配到层级 / Step 1: Assign clients to tiers
        tier_groups = self.assign_clients_to_tiers(contributions)
        
        # 步骤2: 为每个层级计算组内插值 / Step 2: Calculate intra-tier interpolation for each tier
        keep_ratios = {}
        
        for tier_name, client_list in tier_groups.items():
            if not client_list:
                continue
            
            # 提取该层级的所有分数 / Extract all scores in this tier
            tier_scores = [score for _, score in client_list]
            
            # 为每个客户端计算保留率 / Calculate keep ratio for each client
            for client_id, score in client_list:
                keep_ratio = self.calculate_intra_tier_keep_ratio(
                    client_score=score,
                    tier_scores=tier_scores,
                    tier_name=tier_name
                )
                keep_ratios[client_id] = keep_ratio
                
                # 更新客户端的实际层级（如果与传入的不同）
                # Update actual tier if different from input
                if client_id in membership_levels:
                    if membership_levels[client_id] != tier_name:
                        # 这里可以选择是否更新
                        pass
        
        # 统计信息 / Statistics
        self._update_statistics(tier_groups, keep_ratios)
        
        return keep_ratios
    
    def _update_statistics(self, 
                          tier_groups: Dict[str, List[Tuple[int, float]]],
                          keep_ratios: Dict[int, float]) -> None:
        """
        更新统计信息
        Update statistics
        """
        self.round_stats = {
            'tier_distribution': {
                tier: len(clients) for tier, clients in tier_groups.items()
            },
            'tier_keep_ratios': {},
            'overall_stats': {}
        }
        
        # 计算每个层级的平均保留率 / Calculate average keep ratio per tier
        for tier_name, client_list in tier_groups.items():
            if client_list:
                tier_ratios = [keep_ratios[cid] for cid, _ in client_list]
                self.round_stats['tier_keep_ratios'][tier_name] = {
                    'mean': float(np.mean(tier_ratios)),
                    'std': float(np.std(tier_ratios)),
                    'min': float(np.min(tier_ratios)),
                    'max': float(np.max(tier_ratios)),
                    'count': len(tier_ratios)
                }
        
        # 整体统计 / Overall statistics
        if keep_ratios:
            all_ratios = list(keep_ratios.values())
            self.round_stats['overall_stats'] = {
                'mean_keep_ratio': float(np.mean(all_ratios)),
                'std_keep_ratio': float(np.std(all_ratios)),
                'min_keep_ratio': float(np.min(all_ratios)),
                'max_keep_ratio': float(np.max(all_ratios)),
                'mean_sparsity': float(1.0 - np.mean(all_ratios))
            }
    
    def sparsify_gradient_magnitude(self,
                                    gradient: Dict[str, torch.Tensor],
                                    keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        基于幅度的梯度稀疏化（推荐方法）
        Magnitude-based gradient sparsification (recommended method)
        
        保留绝对值最大的 keep_ratio 比例的梯度分量
        这样能保证即使给低质量客户端很少的参数，他们也能获得模型最重要的特征
        
        Retain the top keep_ratio proportion of gradient components by absolute value
        This ensures low-quality clients still get the most important model features
        
        Args:
            gradient: 梯度字典 {param_name: gradient_tensor}
            keep_ratio: 保留率 (0到1之间)
            
        Returns:
            sparse_gradient: 稀疏化后的梯度
        """
        sparse_gradient = {}
        
        for name, grad_tensor in gradient.items():
            if keep_ratio >= 1.0 - self.epsilon:
                # 完全保留 / Fully retain
                sparse_gradient[name] = grad_tensor.clone()
            elif keep_ratio <= self.epsilon:
                # 完全置零 / Fully zero out
                sparse_gradient[name] = torch.zeros_like(grad_tensor)
            else:
                # 部分保留：基于幅度选择 / Partial retention: magnitude-based selection
                flat_grad = grad_tensor.flatten()
                num_params = flat_grad.numel()
                num_keep = max(1, int(num_params * keep_ratio))
                
                # 找到阈值：保留绝对值最大的参数
                # Find threshold: keep parameters with largest absolute values
                abs_grad = torch.abs(flat_grad)
                
                # 使用topk找到阈值 / Use topk to find threshold
                if num_keep >= num_params:
                    threshold = 0.0
                else:
                    # 获取第k大的值作为阈值 / Get k-th largest value as threshold
                    threshold = torch.topk(abs_grad, num_keep, largest=True).values[-1]
                
                # 创建mask：保留大于等于阈值的参数
                # Create mask: retain parameters >= threshold
                mask = (abs_grad >= threshold).float()
                
                # 处理边界情况：确保恰好保留num_keep个参数
                # Handle boundary case: ensure exactly num_keep parameters are kept
                current_kept = mask.sum().item()
                if current_kept > num_keep:
                    # 随机移除多余的参数 / Randomly remove excess parameters
                    kept_indices = torch.where(mask > 0)[0]
                    remove_count = int(current_kept - num_keep)
                    remove_indices = kept_indices[torch.randperm(len(kept_indices))[:remove_count]]
                    mask[remove_indices] = 0.0
                
                # 应用mask / Apply mask
                sparse_flat = flat_grad * mask
                sparse_gradient[name] = sparse_flat.reshape(grad_tensor.shape)
        
        return sparse_gradient
    
    def sparsify_gradient_random(self,
                                 gradient: Dict[str, torch.Tensor],
                                 keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        随机稀疏化梯度
        Random gradient sparsification
        
        随机保留 keep_ratio 比例的梯度分量
        Randomly retain keep_ratio proportion of gradient components
        
        Args:
            gradient: 梯度字典
            keep_ratio: 保留率
            
        Returns:
            sparse_gradient: 稀疏化后的梯度
        """
        sparse_gradient = {}
        
        for name, grad_tensor in gradient.items():
            if keep_ratio >= 1.0 - self.epsilon:
                sparse_gradient[name] = grad_tensor.clone()
            elif keep_ratio <= self.epsilon:
                sparse_gradient[name] = torch.zeros_like(grad_tensor)
            else:
                # 随机mask / Random mask
                mask = (torch.rand_like(grad_tensor) < keep_ratio).float()
                sparse_gradient[name] = grad_tensor * mask
        
        return sparse_gradient
    
    def sparsify_gradient_structured(self,
                                     gradient: Dict[str, torch.Tensor],
                                     keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        结构化稀疏化梯度（按通道/滤波器）
        Structured gradient sparsification (channel/filter-wise)
        
        对卷积层，按滤波器维度稀疏化
        对全连接层，按神经元维度稀疏化
        
        For convolution layers, sparsify by filter dimension
        For fully connected layers, sparsify by neuron dimension
        
        Args:
            gradient: 梯度字典
            keep_ratio: 保留率
            
        Returns:
            sparse_gradient: 稀疏化后的梯度
        """
        sparse_gradient = {}
        
        for name, grad_tensor in gradient.items():
            if keep_ratio >= 1.0 - self.epsilon:
                sparse_gradient[name] = grad_tensor.clone()
            elif keep_ratio <= self.epsilon:
                sparse_gradient[name] = torch.zeros_like(grad_tensor)
            else:
                # 根据张量维度决定稀疏化策略
                # Decide sparsification strategy based on tensor dimensions
                if len(grad_tensor.shape) == 4:  # 卷积层 [out_ch, in_ch, h, w]
                    # 按输出通道稀疏化 / Sparsify by output channel
                    num_filters = grad_tensor.shape[0]
                    num_keep = max(1, int(num_filters * keep_ratio))
                    
                    # 计算每个滤波器的L2范数 / Calculate L2 norm for each filter
                    filter_norms = torch.norm(
                        grad_tensor.reshape(num_filters, -1), 
                        dim=1
                    )
                    
                    # 选择top-k滤波器 / Select top-k filters
                    _, top_indices = torch.topk(filter_norms, num_keep, largest=True)
                    
                    # 创建mask / Create mask
                    mask = torch.zeros_like(grad_tensor)
                    mask[top_indices, :, :, :] = 1.0
                    
                    sparse_gradient[name] = grad_tensor * mask
                
                elif len(grad_tensor.shape) == 2:  # 全连接层 [out, in]
                    # 按输出神经元稀疏化 / Sparsify by output neuron
                    num_neurons = grad_tensor.shape[0]
                    num_keep = max(1, int(num_neurons * keep_ratio))
                    
                    # 计算每个神经元的L2范数 / Calculate L2 norm for each neuron
                    neuron_norms = torch.norm(grad_tensor, dim=1)
                    
                    # 选择top-k神经元 / Select top-k neurons
                    _, top_indices = torch.topk(neuron_norms, num_keep, largest=True)
                    
                    # 创建mask / Create mask
                    mask = torch.zeros_like(grad_tensor)
                    mask[top_indices, :] = 1.0
                    
                    sparse_gradient[name] = grad_tensor * mask
                
                else:
                    # 其他形状：使用magnitude稀疏化 / Other shapes: use magnitude sparsification
                    sparse_gradient[name] = self.sparsify_gradient_magnitude(
                        {name: grad_tensor}, keep_ratio
                    )[name]
        
        return sparse_gradient
    
    def sparsify_gradient(self,
                          gradient: Dict[str, torch.Tensor],
                          keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        根据配置的模式稀疏化梯度
        Sparsify gradient based on configured mode
        
        Args:
            gradient: 梯度字典
            keep_ratio: 保留率
            
        Returns:
            sparse_gradient: 稀疏化后的梯度
        """
        if self.sparsification_mode == 'magnitude':
            return self.sparsify_gradient_magnitude(gradient, keep_ratio)
        elif self.sparsification_mode == 'random':
            return self.sparsify_gradient_random(gradient, keep_ratio)
        elif self.sparsification_mode == 'structured':
            return self.sparsify_gradient_structured(gradient, keep_ratio)
        else:
            # 默认使用magnitude / Default to magnitude
            return self.sparsify_gradient_magnitude(gradient, keep_ratio)
    
    def distribute_sparsified_gradients(self,
                                        global_gradient: Dict[str, torch.Tensor],
                                        membership_levels: Dict[int, str],
                                        contributions: Dict[int, float]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        分发差异化稀疏的梯度给客户端
        Distribute differentially sparsified gradients to clients
        
        核心流程 / Core Workflow:
        1. 计算每个客户端的保留率（使用层级约束和组内插值）
        2. 对全局梯度进行差异化稀疏
        3. 返回每个客户端的稀疏梯度
        
        Calculate keep ratios -> Differential sparsification -> Return sparse gradients
        
        Args:
            global_gradient: 全局聚合梯度
            membership_levels: 客户端会员等级
            contributions: 客户端贡献度
            
        Returns:
            Dict[client_id, sparse_gradient]: 每个客户端的稀疏梯度
        """
        # 步骤1: 计算所有客户端的保留率 / Step 1: Calculate keep ratios for all clients
        keep_ratios = self.calculate_all_keep_ratios(contributions, membership_levels)
        
        # 步骤2: 为每个客户端生成稀疏梯度 / Step 2: Generate sparse gradient for each client
        sparsified_gradients = {}
        
        for client_id, level in membership_levels.items():
            keep_ratio = keep_ratios.get(client_id, 0.5)  # 默认50%保留率 / Default 50%
            
            # 稀疏化梯度 / Sparsify gradient
            sparse_gradient = self.sparsify_gradient(global_gradient, keep_ratio)
            
            sparsified_gradients[client_id] = sparse_gradient
            
            # 记录统计 / Record statistics
            self.sparsification_stats['keep_ratios'].append(keep_ratio)
            self.sparsification_stats['sparsity_rates'].append(1.0 - keep_ratio)
        
        return sparsified_gradients
    
    def get_sparsification_statistics(self) -> Dict:
        """
        获取稀疏化统计信息
        Get sparsification statistics
        
        Returns:
            统计信息字典 / Statistics dictionary
        """
        stats = {
            'round_stats': self.round_stats,
            'overall': {}
        }
        
        if self.sparsification_stats.get('keep_ratios'):
            all_ratios = self.sparsification_stats['keep_ratios']
            stats['overall'] = {
                'avg_keep_ratio': float(np.mean(all_ratios)),
                'std_keep_ratio': float(np.std(all_ratios)),
                'min_keep_ratio': float(np.min(all_ratios)),
                'max_keep_ratio': float(np.max(all_ratios)),
                'avg_sparsity_rate': float(1.0 - np.mean(all_ratios))
            }
        
        # 添加按层级的统计 / Add per-tier statistics
        if 'tier_keep_ratios' in self.round_stats:
            stats['by_level'] = self.round_stats['tier_keep_ratios']
        
        return stats
    
    def print_round_summary(self) -> None:
        """
        打印本轮稀疏化摘要
        Print sparsification summary for current round
        """
        if not self.round_stats:
            return
        
        print(f"\n{'='*60}")
        print(f"Tier-Constrained Gradient Sparsification Summary")
        print(f"层级约束梯度稀疏化摘要")
        print(f"{'='*60}")
        
        # 层级分布 / Tier distribution
        if 'tier_distribution' in self.round_stats:
            print(f"\nTier Distribution / 层级分布:")
            for tier, count in self.round_stats['tier_distribution'].items():
                print(f"  {tier.capitalize():8s}: {count} clients")
        
        # 各层级保留率 / Keep ratios per tier
        if 'tier_keep_ratios' in self.round_stats:
            print(f"\nKeep Ratios by Tier / 各层级保留率:")
            for tier, stats in self.round_stats['tier_keep_ratios'].items():
                print(f"  {tier.capitalize():8s}: "
                      f"Mean={stats['mean']:.3f}, "
                      f"Range=[{stats['min']:.3f}, {stats['max']:.3f}], "
                      f"n={stats['count']}")
        
        # 整体统计 / Overall statistics
        if 'overall_stats' in self.round_stats and self.round_stats['overall_stats']:
            os = self.round_stats['overall_stats']
            print(f"\nOverall Statistics / 整体统计:")
            print(f"  Mean Keep Ratio / 平均保留率: {os['mean_keep_ratio']:.4f}")
            print(f"  Mean Sparsity / 平均稀疏率: {os['mean_sparsity']:.4f}")
            print(f"  Range / 范围: [{os['min_keep_ratio']:.4f}, {os['max_keep_ratio']:.4f}]")
        
        print(f"{'='*60}\n")
    
    def calculate_actual_sparsity(self,
                                  sparse_gradient: Dict[str, torch.Tensor]) -> float:
        """
        计算实际的稀疏率（梯度中零元素的比例）
        Calculate actual sparsity rate (proportion of zero elements in gradient)
        
        Args:
            sparse_gradient: 稀疏梯度
            
        Returns:
            actual_sparsity: 实际稀疏率
        """
        total_params = 0
        zero_params = 0
        
        for grad_tensor in sparse_gradient.values():
            total_params += grad_tensor.numel()
            zero_params += (grad_tensor == 0).sum().item()
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0.0
        return actual_sparsity


# 为了向后兼容，保留原类名的别名
# Alias for backward compatibility
SparsificationDistributor = TierConstrainedGradientDistributor