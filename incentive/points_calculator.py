"""
incentive/points_calculator.py (EMA Enhanced Version)
AMAC贡献度计算器 - 集成EMA平滑
AMAC Contribution Calculator - Integrated with EMA Smoothing
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime


class AMACContributionCalculator:
    """
    AMAC (Adaptive Magnitude-Aware Contribution) 贡献度计算器
    支持EMA平滑的自适应幅度感知贡献度计算
    Adaptive contribution calculation with EMA smoothing support
    """
    
    def __init__(self, T: int = 200, gamma: float = 1.0, epsilon: float = 1e-8,
                 use_ema_smoothing: bool = True, ema_alpha: float = 0.3):
        """
        初始化AMAC计算器 / Initialize AMAC calculator
        
        Args:
            T: 转折点轮次，λ(t)在此轮次达到1 / Transition round where λ(t) reaches 1
            gamma: 幅度得分敏感度参数 / Magnitude score sensitivity parameter
            epsilon: 防止除零的小常数 / Small constant to prevent division by zero
            use_ema_smoothing: 是否对贡献度使用EMA平滑 / Whether to use EMA smoothing for contributions
            ema_alpha: EMA平滑系数 / EMA smoothing factor
        """
        self.T = T
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_ema_smoothing = use_ema_smoothing
        self.ema_alpha = ema_alpha
        
        # 贡献历史记录 / Contribution history
        self.contribution_history = {}  # {client_id: [records]}
        self.gradient_norms_history = []
        
        # EMA平滑后的贡献度 / EMA-smoothed contributions
        self.client_ema_contributions = {}  # {client_id: ema_value}
        
        print(f"🧮 AMACContributionCalculator initialized")
        print(f"   T={T}, gamma={gamma}, EMA_smoothing={use_ema_smoothing}")
        if use_ema_smoothing:
            print(f"   EMA α={ema_alpha}")
    
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
        计算客户端的AMAC贡献度（原始值）/ Calculate client's AMAC contribution (raw value)
        
        公式: Ci(t) = (1 - λ(t)) · Sdir(gi, gagg) + λ(t) · Sconv(gi)
        
        Args:
            client_id: 客户端ID / Client ID
            round_num: 当前轮次 / Current round
            client_gradient: 客户端梯度 / Client gradient
            aggregated_gradient: 聚合梯度 / Aggregated gradient
            all_gradients: 所有客户端的梯度列表 / List of all client gradients
            
        Returns:
            原始AMAC贡献度 / Raw AMAC contribution
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
        
        # 计算原始贡献度 / Calculate raw contribution
        raw_contribution = (1 - lambda_t) * s_dir + lambda_t * s_conv
        
        # 记录贡献历史 / Record contribution history
        if client_id not in self.contribution_history:
            self.contribution_history[client_id] = []
        
        record = {
            'round': round_num,
            'raw_contribution': raw_contribution,
            'lambda_t': lambda_t,
            's_dir': s_dir,
            's_conv': s_conv,
            'timestamp': datetime.now()
        }
        
        # 如果使用EMA平滑，计算并存储EMA值
        if self.use_ema_smoothing:
            smoothed_contribution = self._apply_ema_smoothing(client_id, raw_contribution)
            record['smoothed_contribution'] = smoothed_contribution
        else:
            record['smoothed_contribution'] = raw_contribution
        
        self.contribution_history[client_id].append(record)
        
        # 返回平滑后的值（如果启用EMA）或原始值
        return record['smoothed_contribution']
    
    def _apply_ema_smoothing(self, client_id: int, raw_contribution: float) -> float:
        """
        对贡献度应用EMA平滑 / Apply EMA smoothing to contribution
        
        Args:
            client_id: 客户端ID / Client ID
            raw_contribution: 原始贡献度 / Raw contribution
            
        Returns:
            平滑后的贡献度 / Smoothed contribution
        """
        if client_id not in self.client_ema_contributions:
            # 第一次，EMA初始化为原始值
            self.client_ema_contributions[client_id] = raw_contribution
        else:
            # EMA更新公式
            old_ema = self.client_ema_contributions[client_id]
            self.client_ema_contributions[client_id] = (
                self.ema_alpha * raw_contribution + 
                (1 - self.ema_alpha) * old_ema
            )
        
        return self.client_ema_contributions[client_id]
    
    def get_smoothed_contribution(self, client_id: int) -> float:
        """
        获取客户端的平滑贡献度 / Get client's smoothed contribution
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            平滑后的贡献度 / Smoothed contribution
        """
        if self.use_ema_smoothing:
            return self.client_ema_contributions.get(client_id, 0.0)
        else:
            # 如果未启用EMA，返回最近的原始贡献度
            if client_id in self.contribution_history and self.contribution_history[client_id]:
                return self.contribution_history[client_id][-1]['raw_contribution']
            return 0.0
    
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
        # 收集所有原始贡献度
        raw_contributions = []
        smoothed_contributions = []
        
        for client_history in self.contribution_history.values():
            for record in client_history:
                raw_contributions.append(record['raw_contribution'])
                smoothed_contributions.append(record['smoothed_contribution'])
        
        if not raw_contributions:
            return {}
        
        stats = {
            'raw_contribution_stats': {
                'mean': np.mean(raw_contributions),
                'std': np.std(raw_contributions),
                'min': np.min(raw_contributions),
                'max': np.max(raw_contributions),
            },
            'total_evaluations': len(raw_contributions),
            'num_clients': len(self.contribution_history),
            'ema_enabled': self.use_ema_smoothing
        }
        
        if self.use_ema_smoothing:
            stats['smoothed_contribution_stats'] = {
                'mean': np.mean(smoothed_contributions),
                'std': np.std(smoothed_contributions),
                'min': np.min(smoothed_contributions),
                'max': np.max(smoothed_contributions),
            }
            
            # 计算平滑效果：原始值和平滑值的差异
            if len(raw_contributions) > 1:
                raw_volatility = np.std(np.diff(raw_contributions))
                smoothed_volatility = np.std(np.diff(smoothed_contributions))
                stats['smoothing_effect'] = {
                    'raw_volatility': raw_volatility,
                    'smoothed_volatility': smoothed_volatility,
                    'volatility_reduction': (raw_volatility - smoothed_volatility) / raw_volatility if raw_volatility > 0 else 0
                }
        
        return stats
    
    def calculate_time_slice_contribution(self, client_id: int, 
                                         slice_start: int, 
                                         slice_end: int,
                                         use_smoothed: bool = True) -> float:
        """
        计算时间片内的累积贡献度 / Calculate cumulative contribution within time slice
        
        Args:
            client_id: 客户端ID / Client ID
            slice_start: 时间片开始轮次 / Time slice start round
            slice_end: 时间片结束轮次 / Time slice end round
            use_smoothed: 是否使用平滑值 / Whether to use smoothed values
            
        Returns:
            时间片内的累积贡献度 / Cumulative contribution within time slice
        """
        if client_id not in self.contribution_history:
            return 0.0
        
        slice_contribution = 0.0
        for record in self.contribution_history[client_id]:
            if slice_start <= record['round'] <= slice_end:
                if use_smoothed and 'smoothed_contribution' in record:
                    slice_contribution += record['smoothed_contribution']
                else:
                    slice_contribution += record['raw_contribution']
        
        return slice_contribution
    
    def compare_raw_vs_smoothed(self, client_id: int) -> Dict:
        """
        比较原始贡献度和平滑贡献度 / Compare raw and smoothed contributions
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            比较结果 / Comparison results
        """
        if client_id not in self.contribution_history:
            return {}
        
        history = self.contribution_history[client_id]
        
        rounds = [r['round'] for r in history]
        raw_values = [r['raw_contribution'] for r in history]
        smoothed_values = [r['smoothed_contribution'] for r in history]
        
        return {
            'client_id': client_id,
            'rounds': rounds,
            'raw_contributions': raw_values,
            'smoothed_contributions': smoothed_values,
            'raw_mean': np.mean(raw_values),
            'smoothed_mean': np.mean(smoothed_values),
            'raw_std': np.std(raw_values),
            'smoothed_std': np.std(smoothed_values),
            'correlation': np.corrcoef(raw_values, smoothed_values)[0, 1] if len(raw_values) > 1 else 1.0
        }
    
    def reset_client_history(self, client_id: int) -> None:
        """
        重置客户端历史 / Reset client history
        
        Args:
            client_id: 客户端ID / Client ID
        """
        if client_id in self.contribution_history:
            del self.contribution_history[client_id]
        if client_id in self.client_ema_contributions:
            del self.client_ema_contributions[client_id]
    
    def get_ema_parameters(self) -> Dict:
        """
        获取EMA参数信息 / Get EMA parameters information
        
        Returns:
            EMA参数字典 / EMA parameters dictionary
        """
        equivalent_window = 2 / self.ema_alpha - 1 if self.use_ema_smoothing else 0
        
        return {
            'use_ema_smoothing': self.use_ema_smoothing,
            'ema_alpha': self.ema_alpha,
            'equivalent_window_size': equivalent_window,
            'num_clients_tracked': len(self.client_ema_contributions)
        }