"""
incentive/sparsification_distributor.py - 重构版本
梯度稀疏化分发器 - 对聚合梯度进行差异化稀疏处理
Gradient Sparsification Distributor - Differential Sparsification of Aggregated Gradients
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict
import copy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IncentiveConfig


class SparsificationDistributor:
    """
    梯度稀疏化分发器
    
    核心功能 / Core Functions:
    1. 根据会员等级和贡献度计算稀疏率
    2. 对全局聚合梯度进行差异化稀疏
    3. 分发稀疏化梯度给不同客户端
    
    稀疏化公式 / Sparsification Formula:
        keep_ratio_i = min_keep + (1 - min_keep) * (r_i)^λ
        其中 r_i 是归一化排名，λ 控制曲线形状
    """
    
    def __init__(self, device: torch.device):
        """
        初始化稀疏化分发器
        
        Args:
            device: 计算设备
        """
        self.device = device
        
        # 配置参数
        self.sparsification_mode = IncentiveConfig.SPARSIFICATION_MODE
        self.lambda_coef = IncentiveConfig.LAMBDA
        self.min_keep_ratio = IncentiveConfig.MIN_KEEP_RATIO
        self.max_keep_ratio = IncentiveConfig.MAX_KEEP_RATIO
        self.level_sparsity_ranges = IncentiveConfig.LEVEL_SPARSITY_RANGES
        
        # 统计信息
        self.sparsification_stats = defaultdict(list)
        
        print(f"✓ Gradient Sparsification Distributor initialized")
        print(f"  Mode: {self.sparsification_mode}")
        print(f"  Lambda: {self.lambda_coef}")
        print(f"  Keep ratio range: [{self.min_keep_ratio}, {self.max_keep_ratio}]")
    
    def calculate_keep_ratio(self, 
                            client_id: int,
                            membership_level: str,
                            contributions: Dict[int, float]) -> float:
        """
        计算客户端的保留率（1 - 稀疏率）
        
        基于会员等级和贡献度排名的双重控制
        
        Args:
            client_id: 客户端ID
            membership_level: 会员等级
            contributions: 所有客户端的贡献度
        
        Returns:
            keep_ratio: 保留率 (0到1之间)
        """
        # 获取等级对应的稀疏率范围
        if membership_level not in self.level_sparsity_ranges:
            membership_level = 'bronze'
        
        min_sparsity, max_sparsity = self.level_sparsity_ranges[membership_level]
        
        # 计算贡献度排名（归一化到0-1）
        if contributions and len(contributions) > 1:
            sorted_clients = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            rank_dict = {cid: idx for idx, (cid, _) in enumerate(sorted_clients)}
            
            if client_id in rank_dict:
                rank = rank_dict[client_id]
                normalized_rank = 1.0 - (rank / (len(contributions) - 1))  # 高贡献者 → 1
            else:
                normalized_rank = 0.5
        else:
            normalized_rank = 0.5
        
        # 使用幂律函数计算保留率
        # keep_ratio = min_keep + (1 - min_keep) * (normalized_rank)^λ
        rank_factor = np.power(normalized_rank, self.lambda_coef)
        
        # 在等级范围内调整
        level_min_keep = 1.0 - max_sparsity
        level_max_keep = 1.0 - min_sparsity
        
        keep_ratio = level_min_keep + (level_max_keep - level_min_keep) * rank_factor
        
        # 确保在全局范围内
        keep_ratio = np.clip(keep_ratio, self.min_keep_ratio, self.max_keep_ratio)
        
        return float(keep_ratio)
    
    def sparsify_gradient_magnitude(self, 
                                    gradient: Dict[str, torch.Tensor],
                                    keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        基于梯度幅度的稀疏化
        
        保留绝对值最大的 keep_ratio 比例的梯度分量
        
        Args:
            gradient: 梯度字典
            keep_ratio: 保留率
        
        Returns:
            sparse_gradient: 稀疏化后的梯度
        """
        sparse_gradient = {}
        
        for name, grad_tensor in gradient.items():
            if keep_ratio >= 1.0:
                # 完全保留
                sparse_gradient[name] = grad_tensor.clone()
            elif keep_ratio <= 0.0:
                # 完全置零
                sparse_gradient[name] = torch.zeros_like(grad_tensor)
            else:
                # 部分保留
                flat_grad = grad_tensor.flatten()
                num_params = flat_grad.numel()
                num_keep = max(1, int(num_params * keep_ratio))
                
                # 找到阈值
                abs_grad = torch.abs(flat_grad)
                threshold = torch.topk(abs_grad, num_keep, largest=True).values[-1]
                
                # 创建mask
                mask = (abs_grad >= threshold).float()
                
                # 应用mask
                sparse_flat = flat_grad * mask
                sparse_gradient[name] = sparse_flat.reshape(grad_tensor.shape)
        
        return sparse_gradient
    
    def sparsify_gradient_random(self,
                                gradient: Dict[str, torch.Tensor],
                                keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        随机稀疏化梯度
        
        随机保留 keep_ratio 比例的梯度分量
        
        Args:
            gradient: 梯度字典
            keep_ratio: 保留率
        
        Returns:
            sparse_gradient: 稀疏化后的梯度
        """
        sparse_gradient = {}
        
        for name, grad_tensor in gradient.items():
            if keep_ratio >= 1.0:
                sparse_gradient[name] = grad_tensor.clone()
            elif keep_ratio <= 0.0:
                sparse_gradient[name] = torch.zeros_like(grad_tensor)
            else:
                # 随机mask
                mask = (torch.rand_like(grad_tensor) < keep_ratio).float()
                sparse_gradient[name] = grad_tensor * mask
        
        return sparse_gradient
    
    def sparsify_gradient_structured(self,
                                    gradient: Dict[str, torch.Tensor],
                                    keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        结构化稀疏化梯度（按通道/滤波器）
        
        对卷积层，按滤波器维度稀疏化
        对全连接层，按神经元维度稀疏化
        
        Args:
            gradient: 梯度字典
            keep_ratio: 保留率
        
        Returns:
            sparse_gradient: 稀疏化后的梯度
        """
        sparse_gradient = {}
        
        for name, grad_tensor in gradient.items():
            if keep_ratio >= 1.0:
                sparse_gradient[name] = grad_tensor.clone()
            elif keep_ratio <= 0.0:
                sparse_gradient[name] = torch.zeros_like(grad_tensor)
            else:
                # 根据张量维度决定稀疏化策略
                if len(grad_tensor.shape) == 4:  # 卷积层 [out_ch, in_ch, h, w]
                    # 按输出通道稀疏化
                    num_filters = grad_tensor.shape[0]
                    num_keep = max(1, int(num_filters * keep_ratio))
                    
                    # 计算每个滤波器的L2范数
                    filter_norms = torch.norm(
                        grad_tensor.reshape(num_filters, -1), 
                        dim=1
                    )
                    
                    # 选择top-k滤波器
                    _, top_indices = torch.topk(filter_norms, num_keep, largest=True)
                    
                    # 创建mask
                    mask = torch.zeros_like(grad_tensor)
                    mask[top_indices, :, :, :] = 1.0
                    
                    sparse_gradient[name] = grad_tensor * mask
                
                elif len(grad_tensor.shape) == 2:  # 全连接层 [out, in]
                    # 按输出神经元稀疏化
                    num_neurons = grad_tensor.shape[0]
                    num_keep = max(1, int(num_neurons * keep_ratio))
                    
                    # 计算每个神经元的L2范数
                    neuron_norms = torch.norm(grad_tensor, dim=1)
                    
                    # 选择top-k神经元
                    _, top_indices = torch.topk(neuron_norms, num_keep, largest=True)
                    
                    # 创建mask
                    mask = torch.zeros_like(grad_tensor)
                    mask[top_indices, :] = 1.0
                    
                    sparse_gradient[name] = grad_tensor * mask
                
                else:
                    # 其他形状（如BN参数），使用magnitude稀疏化
                    sparse_gradient[name] = self.sparsify_gradient_magnitude(
                        {name: grad_tensor}, keep_ratio
                    )[name]
        
        return sparse_gradient
    
    def sparsify_gradient(self,
                         gradient: Dict[str, torch.Tensor],
                         keep_ratio: float) -> Dict[str, torch.Tensor]:
        """
        根据配置的模式稀疏化梯度
        
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
            raise ValueError(f"Unknown sparsification mode: {self.sparsification_mode}")
    
    def distribute_sparsified_gradients(self,
                                       global_gradient: Dict[str, torch.Tensor],
                                       membership_levels: Dict[int, str],
                                       contributions: Dict[int, float]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        分发差异化稀疏的梯度给客户端
        
        核心流程 / Core Workflow:
        1. 计算每个客户端的保留率
        2. 对全局梯度进行差异化稀疏
        3. 返回每个客户端的稀疏梯度
        
        Args:
            global_gradient: 全局聚合梯度
            membership_levels: 客户端会员等级
            contributions: 客户端贡献度
        
        Returns:
            Dict[client_id, sparse_gradient]: 每个客户端的稀疏梯度
        """
        sparsified_gradients = {}
        
        # 统计信息
        level_stats = defaultdict(lambda: {'keep_ratios': [], 'count': 0})
        
        for client_id, level in membership_levels.items():
            # 计算保留率
            keep_ratio = self.calculate_keep_ratio(client_id, level, contributions)
            
            # 稀疏化梯度
            sparse_gradient = self.sparsify_gradient(global_gradient, keep_ratio)
            
            sparsified_gradients[client_id] = sparse_gradient
            
            # 统计
            sparsity_rate = 1.0 - keep_ratio
            level_stats[level]['keep_ratios'].append(keep_ratio)
            level_stats[level]['count'] += 1
            
            self.sparsification_stats['keep_ratios'].append(keep_ratio)
            self.sparsification_stats['sparsity_rates'].append(sparsity_rate)
        
        # 记录本轮统计
        self.sparsification_stats['round_level_stats'] = level_stats
        
        return sparsified_gradients
    
    def get_sparsification_statistics(self) -> Dict:
        """获取稀疏化统计信息"""
        if not self.sparsification_stats.get('keep_ratios'):
            return {}
        
        level_stats = self.sparsification_stats.get('round_level_stats', {})
        
        stats = {
            'overall': {
                'avg_keep_ratio': float(np.mean(self.sparsification_stats['keep_ratios'])),
                'avg_sparsity_rate': float(np.mean(self.sparsification_stats['sparsity_rates'])),
                'min_keep_ratio': float(np.min(self.sparsification_stats['keep_ratios'])),
                'max_keep_ratio': float(np.max(self.sparsification_stats['keep_ratios']))
            },
            'by_level': {}
        }
        
        for level, level_data in level_stats.items():
            if level_data['keep_ratios']:
                stats['by_level'][level] = {
                    'avg_keep_ratio': float(np.mean(level_data['keep_ratios'])),
                    'avg_sparsity_rate': 1.0 - float(np.mean(level_data['keep_ratios'])),
                    'count': level_data['count']
                }
        
        return stats
    
    def calculate_actual_sparsity(self, 
                                 sparse_gradient: Dict[str, torch.Tensor]) -> float:
        """
        计算实际的稀疏率（梯度中零元素的比例）
        
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