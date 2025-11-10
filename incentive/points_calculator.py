"""
incentive/points_calculator.py
积分计算器 - 使用CGSV方法 / Points Calculator - Using CGSV Method
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import copy


class CGSVCalculator:
    """
    CGSV (Cosine Gradient Shapley Value) 计算器
    CGSV Calculator for contribution evaluation
    基于余弦相似度的梯度Shapley值计算客户端贡献度
    Calculate client contributions based on cosine similarity of gradients
    """
    
    def __init__(self):
        """初始化CGSV计算器 / Initialize CGSV calculator"""
        self.gradient_cache = {}  # 缓存梯度信息 / Cache gradient information
        self.contribution_history = {}  # 贡献历史 / Contribution history
        
    def compute_gradient_vector(self, model_weights: Dict, 
                               prev_weights: Dict) -> np.ndarray:
        """
        计算模型更新的梯度向量 / Compute gradient vector from model updates
        
        Args:
            model_weights: 更新后的模型权重 / Updated model weights
            prev_weights: 更新前的模型权重 / Previous model weights
            
        Returns:
            梯度向量 / Gradient vector
        """
        gradients = []
        
        for key in model_weights.keys():
            if key in prev_weights:
                # 计算参数差异作为梯度 / Calculate parameter difference as gradient
                grad = model_weights[key] - prev_weights[key]
                # 展平并转换为numpy数组 / Flatten and convert to numpy array
                gradients.append(grad.cpu().numpy().flatten())
        
        # 连接所有梯度为一个向量 / Concatenate all gradients into one vector
        gradient_vector = np.concatenate(gradients)
        
        return gradient_vector
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度 / Calculate cosine similarity between two vectors
        
        Args:
            vec1: 向量1 / Vector 1
            vec2: 向量2 / Vector 2
            
        Returns:
            余弦相似度 / Cosine similarity
        """
        # 避免除零错误 / Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 计算余弦相似度 / Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # 确保在[-1, 1]范围内 / Ensure within [-1, 1] range
        return np.clip(similarity, -1.0, 1.0)
    
    def calculate_cgsv_contributions(self, client_gradients: Dict[int, np.ndarray],
                                    global_gradient: np.ndarray) -> Dict[int, float]:
        """
        计算基于CGSV的客户端贡献度 / Calculate CGSV-based client contributions
        
        Args:
            client_gradients: 客户端梯度字典 / Client gradients dictionary
            global_gradient: 全局梯度 / Global gradient
            
        Returns:
            客户端贡献度字典 / Client contribution dictionary
        """
        contributions = {}
        
        # 计算每个客户端梯度与全局梯度的余弦相似度 / Calculate cosine similarity for each client
        for client_id, client_grad in client_gradients.items():
            similarity = self.cosine_similarity(client_grad, global_gradient)
            
            # 将相似度转换为贡献度分数 (映射到[0,1]) / Convert similarity to contribution score
            # 使用(similarity + 1) / 2 将[-1, 1]映射到[0, 1]
            contribution_score = (similarity + 1) / 2
            
            contributions[client_id] = contribution_score
        
        # 归一化贡献度 / Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            for client_id in contributions:
                contributions[client_id] /= total_contribution
        
        return contributions
    
    def calculate_shapley_value(self, client_gradients: Dict[int, np.ndarray],
                               target_gradient: np.ndarray,
                               client_id: int,
                               num_samples: int = 10) -> float:
        """
        计算单个客户端的Shapley值 / Calculate Shapley value for a single client
        使用蒙特卡洛采样近似计算 / Use Monte Carlo sampling for approximation
        
        Args:
            client_gradients: 所有客户端的梯度 / All client gradients
            target_gradient: 目标梯度 / Target gradient
            client_id: 要计算的客户端ID / Client ID to calculate
            num_samples: 采样次数 / Number of samples
            
        Returns:
            Shapley值 / Shapley value
        """
        other_clients = [cid for cid in client_gradients.keys() if cid != client_id]
        marginal_contributions = []
        
        for _ in range(num_samples):
            # 随机排列其他客户端 / Random permutation of other clients
            np.random.shuffle(other_clients)
            
            # 随机选择子集大小 / Randomly choose subset size
            subset_size = np.random.randint(0, len(other_clients) + 1)
            subset = other_clients[:subset_size]
            
            # 计算加入该客户端前的联盟值 / Calculate coalition value before adding client
            if subset:
                coalition_grad = np.mean([client_gradients[cid] for cid in subset], axis=0)
                value_without = self.cosine_similarity(coalition_grad, target_gradient)
            else:
                value_without = 0
            
            # 计算加入该客户端后的联盟值 / Calculate coalition value after adding client
            subset_with = subset + [client_id]
            coalition_grad_with = np.mean([client_gradients[cid] for cid in subset_with], axis=0)
            value_with = self.cosine_similarity(coalition_grad_with, target_gradient)
            
            # 边际贡献 / Marginal contribution
            marginal_contributions.append(value_with - value_without)
        
        # 返回平均边际贡献作为Shapley值 / Return average marginal contribution as Shapley value
        return np.mean(marginal_contributions)


class PointsCalculator:
    """
    积分计算器 / Points Calculator
    基于CGSV贡献度计算客户端积分
    Calculate client points based on CGSV contributions
    """
    
    def __init__(self):
        """初始化积分计算器 / Initialize points calculator"""
        self.cgsv_calculator = CGSVCalculator()
        self.points_history = {}
        self.contribution_history = {}
        
        # 积分计算参数 / Points calculation parameters
        self.base_points = 1000  # 基础积分 / Base points
        self.max_bonus_multiplier = 3.0  # 最大奖励倍数 / Maximum bonus multiplier
        
    def calculate_points_with_cgsv(self, client_gradients: Dict[int, np.ndarray],
                                  global_gradient: np.ndarray,
                                  client_infos: Dict[int, Dict]) -> Dict[int, float]:
        """
        使用CGSV计算所有客户端的积分 / Calculate points for all clients using CGSV
        
        Args:
            client_gradients: 客户端梯度字典 / Client gradients dictionary
            global_gradient: 全局梯度 / Global gradient
            client_infos: 客户端信息字典 / Client information dictionary
            
        Returns:
            客户端积分字典 / Client points dictionary
        """
        # 计算CGSV贡献度 / Calculate CGSV contributions
        contributions = self.cgsv_calculator.calculate_cgsv_contributions(
            client_gradients, global_gradient
        )
        
        points = {}
        for client_id, contribution in contributions.items():
            # 基础积分 = 贡献度 * 基础积分值 / Base points = contribution * base points value
            base_points = contribution * self.base_points
            
            # 考虑数据量和计算时间的额外奖励 / Consider data size and computation time bonuses
            info = client_infos.get(client_id, {})
            data_bonus = min(info.get('num_samples', 0) / 1000, 1.0) * 200
            time_bonus = min(info.get('computation_time', 0) / 10, 1.0) * 100
            
            # 等级倍数 / Level multiplier
            level_multiplier = info.get('level_multiplier', 1.0)
            
            # 总积分 / Total points
            total_points = (base_points + data_bonus + time_bonus) * level_multiplier
            
            points[client_id] = total_points
            
            # 记录贡献度历史 / Record contribution history
            if client_id not in self.contribution_history:
                self.contribution_history[client_id] = []
            self.contribution_history[client_id].append(contribution)
        
        return points
    
    def calculate_gradient_based_reward(self, contribution: float) -> float:
        """
        根据贡献度计算梯度奖励 / Calculate gradient-based reward from contribution
        
        Args:
            contribution: CGSV贡献度 (0-1) / CGSV contribution (0-1)
            
        Returns:
            奖励倍数 / Reward multiplier
        """
        # 使用非线性函数映射贡献度到奖励 / Use non-linear function to map contribution to reward
        # 贡献度越高，奖励增长越快 / Higher contribution leads to faster reward growth
        reward_multiplier = 1.0 + (self.max_bonus_multiplier - 1.0) * (contribution ** 2)
        
        return reward_multiplier
    
    def record_points(self, client_id: int, round_num: int, points: float) -> None:
        """
        记录积分历史 / Record points history
        
        Args:
            client_id: 客户端ID / Client ID
            round_num: 轮次 / Round number
            points: 获得的积分 / Points earned
        """
        if client_id not in self.points_history:
            self.points_history[client_id] = []
        
        self.points_history[client_id].append({
            'round': round_num,
            'points': points,
            'timestamp': datetime.now()
        })
    
    def get_contribution_stats(self, client_id: int) -> Dict:
        """
        获取客户端贡献统计 / Get client contribution statistics
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            贡献统计信息 / Contribution statistics
        """
        if client_id not in self.contribution_history:
            return {
                'avg_contribution': 0,
                'max_contribution': 0,
                'min_contribution': 0,
                'total_rounds': 0
            }
        
        contributions = self.contribution_history[client_id]
        
        return {
            'avg_contribution': np.mean(contributions),
            'max_contribution': np.max(contributions),
            'min_contribution': np.min(contributions),
            'total_rounds': len(contributions),
            'recent_contribution': contributions[-1] if contributions else 0
        }