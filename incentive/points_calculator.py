"""
incentive/points_calculator.py
AMAC贡献度计算器 / AMAC Contribution Calculator
基于自适应幅度感知的贡献度量 / Based on Adaptive Magnitude-Aware Contribution
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime


class AMACContributionCalculator:
    """
    AMAC (Adaptive Magnitude-Aware Contribution) 贡献度计算器
    自适应幅度感知贡献度计算，解决CGSV的贡献递减问题
    Adaptive contribution calculation addressing the diminishing returns problem of CGSV
    """
    
    def __init__(self, T: int = 200, gamma: float = 1.0, epsilon: float = 1e-8):
        """
        初始化AMAC计算器 / Initialize AMAC calculator
        
        Args:
            T: 转折点轮次，λ(t)在此轮次达到1 / Transition round where λ(t) reaches 1
            gamma: 幅度得分敏感度参数 / Magnitude score sensitivity parameter
            epsilon: 防止除零的小常数 / Small constant to prevent division by zero
        """
        self.T = T
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 贡献历史记录 / Contribution history
        self.contribution_history = {}
        self.gradient_norms_history = []
        
    def calculate_adaptive_weight(self, round_num: int) -> float:
        """
        计算自适应权重λ(t) / Calculate adaptive weight λ(t)
        
        在训练初期(t→0)时λ(t)→0，主要关注方向
        在训练后期(t≥T)时λ(t)→1，主要关注收敛性
        
        Args:
            round_num: 当前训练轮次 / Current training round
            
        Returns:
            自适应权重 / Adaptive weight
        """
        return min(1.0, round_num / self.T)
    
    def calculate_direction_score(self, gi: torch.Tensor, gagg: torch.Tensor) -> float:
        """
        计算方向贡献分Sdir / Calculate direction contribution score
        使用平滑的余弦相似度，增加鲁棒性
        Using smoothed cosine similarity for robustness
        
        Args:
            gi: 客户端i的梯度 / Gradient of client i
            gagg: 聚合梯度 / Aggregated gradient
            
        Returns:
            方向贡献分 / Direction contribution score
        """
        # 将梯度展平为一维向量 / Flatten gradients to 1D vectors
        gi_flat = torch.cat([g.flatten() for g in gi.values()]) if isinstance(gi, dict) else gi.flatten()
        gagg_flat = torch.cat([g.flatten() for g in gagg.values()]) if isinstance(gagg, dict) else gagg.flatten()
        
        # 计算平滑余弦相似度 / Calculate smoothed cosine similarity
        dot_product = torch.dot(gi_flat, gagg_flat).item()
        norm_gi = torch.norm(gi_flat).item() + self.epsilon
        norm_gagg = torch.norm(gagg_flat).item() + self.epsilon
        
        cosine_sim = dot_product / (norm_gi * norm_gagg)
        
        # 确保非负（排除完全相反方向的贡献）/ Ensure non-negative (exclude opposite directions)
        return max(0.0, cosine_sim)
    
    def calculate_convergence_score(self, gi: torch.Tensor, G_bar: float) -> float:
        """
        计算收敛贡献分Sconv / Calculate convergence contribution score
        梯度幅度越小，收敛性越好，得分越高
        Smaller gradient magnitude indicates better convergence, higher score
        
        Args:
            gi: 客户端i的梯度 / Gradient of client i
            G_bar: 所有客户端梯度幅度的均值 / Mean gradient magnitude of all clients
            
        Returns:
            收敛贡献分 / Convergence contribution score
        """
        # 计算梯度幅度 / Calculate gradient magnitude
        gi_flat = torch.cat([g.flatten() for g in gi.values()]) if isinstance(gi, dict) else gi.flatten()
        norm_gi = torch.norm(gi_flat).item()
        
        # 使用归一化的指数衰减函数 / Use normalized exponential decay function
        normalized_norm = norm_gi / (G_bar + self.epsilon)
        score = np.exp(-self.gamma * normalized_norm)
        
        return score
    
    def calculate_contribution(self, client_id: int, round_num: int,
                             client_gradient: Dict[str, torch.Tensor],
                             aggregated_gradient: Dict[str, torch.Tensor],
                             all_gradients: List[Dict[str, torch.Tensor]]) -> float:
        """
        计算客户端的AMAC贡献度 / Calculate client's AMAC contribution
        
        公式: Ci(t) = (1 - λ(t)) · Sdir(gi, gagg) + λ(t) · Sconv(gi)
        
        Args:
            client_id: 客户端ID / Client ID
            round_num: 当前轮次 / Current round
            client_gradient: 客户端梯度 / Client gradient
            aggregated_gradient: 聚合梯度 / Aggregated gradient
            all_gradients: 所有客户端的梯度列表 / List of all client gradients
            
        Returns:
            AMAC贡献度 / AMAC contribution
        """
        # 计算自适应权重 / Calculate adaptive weight
        lambda_t = self.calculate_adaptive_weight(round_num)
        
        # 计算方向贡献分 / Calculate direction score
        s_dir = self.calculate_direction_score(client_gradient, aggregated_gradient)
        
        # 计算所有客户端的梯度幅度均值 / Calculate mean gradient magnitude
        gradient_norms = []
        for grad in all_gradients:
            grad_flat = torch.cat([g.flatten() for g in grad.values()])
            gradient_norms.append(torch.norm(grad_flat).item())
        G_bar = np.mean(gradient_norms)
        
        # 计算收敛贡献分 / Calculate convergence score
        s_conv = self.calculate_convergence_score(client_gradient, G_bar)
        
        # 计算最终贡献度 / Calculate final contribution
        contribution = (1 - lambda_t) * s_dir + lambda_t * s_conv
        
        # 记录贡献历史 / Record contribution history
        if client_id not in self.contribution_history:
            self.contribution_history[client_id] = []
        
        self.contribution_history[client_id].append({
            'round': round_num,
            'contribution': contribution,
            'lambda_t': lambda_t,
            's_dir': s_dir,
            's_conv': s_conv,
            'timestamp': datetime.now()
        })
        
        return contribution
    
    def get_client_contribution_history(self, client_id: int) -> List[Dict]:
        """
        获取客户端贡献历史 / Get client contribution history
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            贡献历史列表 / Contribution history list
        """
        return self.contribution_history.get(client_id, [])
    
    def get_contribution_statistics(self) -> Dict:
        """
        获取贡献度统计信息 / Get contribution statistics
        
        Returns:
            统计信息字典 / Statistics dictionary
        """
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
            'total_evaluations': len(all_contributions)
        }
    
    def calculate_time_slice_contribution(self, client_id: int, 
                                         slice_start: int, 
                                         slice_end: int) -> float:
        """
        计算时间片内的累积贡献度 / Calculate cumulative contribution within time slice
        
        Args:
            client_id: 客户端ID / Client ID
            slice_start: 时间片开始轮次 / Time slice start round
            slice_end: 时间片结束轮次 / Time slice end round
            
        Returns:
            时间片内的累积贡献度 / Cumulative contribution within time slice
        """
        if client_id not in self.contribution_history:
            return 0.0
        
        slice_contribution = 0.0
        for record in self.contribution_history[client_id]:
            if slice_start <= record['round'] <= slice_end:
                slice_contribution += record['contribution']
        
        return slice_contribution