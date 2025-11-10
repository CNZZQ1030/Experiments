"""
incentive/membership.py
会员等级系统 / Membership Level System
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta


class MembershipSystem:
    """
    会员等级管理系统 / Membership Level Management System
    基于海底捞模式的会员等级和权益管理
    Based on Haidilao model membership level and benefits management
    """
    
    def __init__(self, level_thresholds: Dict[str, int], 
                 level_multipliers: Dict[str, float]):
        """
        初始化会员系统 / Initialize membership system
        
        Args:
            level_thresholds: 等级阈值 / Level thresholds
            level_multipliers: 等级倍数 / Level multipliers
        """
        self.level_thresholds = level_thresholds
        self.level_multipliers = level_multipliers
        self.levels = ['bronze', 'silver', 'gold', 'diamond']
        
        # 客户端会员信息 / Client membership information
        self.client_memberships = {}
        
    def initialize_client(self, client_id: int) -> None:
        """
        初始化客户端会员信息 / Initialize client membership information
        
        Args:
            client_id: 客户端ID / Client ID
        """
        self.client_memberships[client_id] = {
            'level': 'bronze',
            'total_points': 0,
            'active_points': 0,  # 有效期内的积分 / Points within validity period
            'points_history': [],  # 积分历史 / Points history
            'level_history': [],  # 等级历史 / Level history
            'join_time': datetime.now()
        }
    
    def update_membership_level(self, client_id: int, points: float) -> str:
        """
        更新会员等级 / Update membership level
        
        Args:
            client_id: 客户端ID / Client ID
            points: 当前积分 / Current points
            
        Returns:
            新的会员等级 / New membership level
        """
        if client_id not in self.client_memberships:
            self.initialize_client(client_id)
        
        # 确定等级 / Determine level
        new_level = 'bronze'
        for level in reversed(self.levels):
            if points >= self.level_thresholds[level]:
                new_level = level
                break
        
        # 更新信息 / Update information
        membership = self.client_memberships[client_id]
        old_level = membership['level']
        membership['level'] = new_level
        membership['total_points'] = points
        
        # 记录等级变化 / Record level change
        if old_level != new_level:
            membership['level_history'].append({
                'timestamp': datetime.now(),
                'old_level': old_level,
                'new_level': new_level,
                'points': points
            })
            print(f"Client {client_id} level changed: {old_level} -> {new_level}")
        
        return new_level
    
    def get_level_benefits(self, level: str) -> Dict:
        """
        获取等级权益 / Get level benefits
        
        Args:
            level: 会员等级 / Membership level
            
        Returns:
            权益信息 / Benefits information
        """
        multiplier = self.level_multipliers.get(level, 1.0)
        
        benefits = {
            'points_multiplier': multiplier,
            'priority_selection': level in ['gold', 'diamond'],
            'extra_rewards': level == 'diamond',
            'aggregation_weight_bonus': multiplier,
            'resource_allocation_priority': self.levels.index(level) + 1
        }
        
        return benefits
    
    def get_client_membership_info(self, client_id: int) -> Dict:
        """
        获取客户端会员信息 / Get client membership information
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            会员信息 / Membership information
        """
        if client_id not in self.client_memberships:
            self.initialize_client(client_id)
        
        return self.client_memberships[client_id].copy()
    
    def get_membership_statistics(self) -> Dict:
        """
        获取会员统计信息 / Get membership statistics
        
        Returns:
            统计信息 / Statistics information
        """
        level_counts = {level: 0 for level in self.levels}
        total_points = 0
        avg_points = 0
        
        for membership in self.client_memberships.values():
            level_counts[membership['level']] += 1
            total_points += membership['total_points']
        
        num_clients = len(self.client_memberships)
        if num_clients > 0:
            avg_points = total_points / num_clients
        
        return {
            'total_clients': num_clients,
            'level_distribution': level_counts,
            'total_points': total_points,
            'average_points': avg_points
        }