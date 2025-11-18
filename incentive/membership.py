"""
incentive/membership.py (Updated with Relative Ranking)
会员等级系统 - 基于相对排名 / Membership Level System - Based on Relative Ranking
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta


class MembershipSystem:
    """
    会员等级管理系统 - 基于相对排名 / Membership Level Management System - Relative Ranking
    
    核心改进：使用相对排名代替绝对积分阈值
    Core improvement: Use relative ranking instead of absolute point thresholds
    
    排名规则 / Ranking Rules:
    - 钻石级 Diamond: Top 25% (前25%)
    - 金级 Gold: 25%-50%
    - 银级 Silver: 50%-75%  
    - 铜级 Bronze: 75%-100% (后25%)
    """
    
    def __init__(self, level_multipliers: Dict[str, float],
                 ranking_percentiles: Dict[str, float] = None):
        """
        初始化会员系统 / Initialize membership system
        
        Args:
            level_multipliers: 等级倍数 / Level multipliers
            ranking_percentiles: 等级排名百分位 / Level ranking percentiles
        """
        self.level_multipliers = level_multipliers
        self.levels = ['bronze', 'silver', 'gold', 'diamond']
        
        # 使用相对排名百分位 / Use relative ranking percentiles
        # 格式：{等级: 最低排名百分位}
        # Format: {level: minimum ranking percentile}
        if ranking_percentiles is None:
            self.ranking_percentiles = {
                'diamond': 0.90,  # 前10% (top 10%)
                'gold': 0.70,     # 前30% (top 30%)
                'silver': 0.40,   # 前60% (top 60%)
                'bronze': 0.00    # 所有 (all)
            }
        else:
            self.ranking_percentiles = ranking_percentiles
        
        # 客户端会员信息 / Client membership information
        self.client_memberships = {}
        
        print(f"MembershipSystem initialized with RELATIVE RANKING:")
        print(f"  Diamond: Top {(1-self.ranking_percentiles['diamond'])*100:.0f}%")
        print(f"  Gold: Top {(1-self.ranking_percentiles['gold'])*100:.0f}%")
        print(f"  Silver: Top {(1-self.ranking_percentiles['silver'])*100:.0f}%")
        print(f"  Bronze: Remaining")
        
    def initialize_client(self, client_id: int) -> None:
        """
        初始化客户端会员信息 / Initialize client membership information
        
        Args:
            client_id: 客户端ID / Client ID
        """
        self.client_memberships[client_id] = {
            'level': 'bronze',
            'total_points': 0,
            'active_points': 0,
            'ranking_percentile': 0.0,  # 当前排名百分位
            'rank': None,  # 当前排名
            'points_history': [],
            'level_history': [],
            'join_time': datetime.now()
        }
    
    def update_all_memberships_by_ranking(self, client_points: Dict[int, float]) -> Dict[int, str]:
        """
        基于相对排名更新所有客户端的会员等级 / Update all memberships by relative ranking
        
        这是核心改进方法！/ This is the core improvement method!
        
        Args:
            client_points: 所有客户端的积分 {client_id: points}
            
        Returns:
            更新后的等级字典 {client_id: new_level}
        """
        if not client_points:
            return {}
        
        # 按积分排序客户端 / Sort clients by points
        sorted_clients = sorted(client_points.items(), key=lambda x: x[1], reverse=True)
        total_clients = len(sorted_clients)
        
        new_levels = {}
        
        for rank, (client_id, points) in enumerate(sorted_clients, start=1):
            # 计算排名百分位 (0=最低, 1=最高)
            # Calculate ranking percentile (0=lowest, 1=highest)
            percentile = (total_clients - rank) / total_clients if total_clients > 1 else 1.0
            
            # 根据百分位确定等级 / Determine level by percentile
            if percentile >= self.ranking_percentiles['diamond']:
                new_level = 'diamond'
            elif percentile >= self.ranking_percentiles['gold']:
                new_level = 'gold'
            elif percentile >= self.ranking_percentiles['silver']:
                new_level = 'silver'
            else:
                new_level = 'bronze'
            
            # 更新客户端信息 / Update client info
            if client_id not in self.client_memberships:
                self.initialize_client(client_id)
            
            membership = self.client_memberships[client_id]
            old_level = membership['level']
            membership['level'] = new_level
            membership['total_points'] = points
            membership['active_points'] = points
            membership['ranking_percentile'] = percentile
            membership['rank'] = rank
            
            new_levels[client_id] = new_level
            
            # 记录等级变化 / Record level change
            if old_level != new_level:
                membership['level_history'].append({
                    'timestamp': datetime.now(),
                    'old_level': old_level,
                    'new_level': new_level,
                    'points': points,
                    'rank': rank,
                    'percentile': percentile
                })
                
                if client_id < 5:  # 只打印前5个客户端的变化
                    print(f"Client {client_id}: {old_level} → {new_level} "
                          f"(Rank: {rank}/{total_clients}, Percentile: {percentile:.2%})")
        
        return new_levels
    
    def update_membership_level(self, client_id: int, points: float) -> str:
        """
        更新单个客户端会员等级（兼容性方法，建议使用update_all_memberships_by_ranking）
        Update single client membership level (compatibility method)
        
        注意：这个方法无法实现相对排名，仅用于向后兼容
        Note: This method cannot implement relative ranking, only for backward compatibility
        """
        if client_id not in self.client_memberships:
            self.initialize_client(client_id)
        
        membership = self.client_memberships[client_id]
        membership['total_points'] = points
        
        # 返回当前等级（实际等级由update_all_memberships_by_ranking决定）
        return membership['level']
    
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
        
        ranks_by_level = {level: [] for level in self.levels}
        
        for membership in self.client_memberships.values():
            level_counts[membership['level']] += 1
            total_points += membership['total_points']
            
            if membership['rank'] is not None:
                ranks_by_level[membership['level']].append(membership['rank'])
        
        num_clients = len(self.client_memberships)
        if num_clients > 0:
            avg_points = total_points / num_clients
        
        # 计算每个等级的平均排名
        avg_ranks = {}
        for level, ranks in ranks_by_level.items():
            avg_ranks[level] = np.mean(ranks) if ranks else None
        
        return {
            'total_clients': num_clients,
            'level_distribution': level_counts,
            'level_percentages': {
                level: count / num_clients * 100 if num_clients > 0 else 0
                for level, count in level_counts.items()
            },
            'total_points': total_points,
            'average_points': avg_points,
            'average_ranks_by_level': avg_ranks
        }
    
    def print_membership_distribution(self) -> None:
        """打印会员等级分布 / Print membership distribution"""
        stats = self.get_membership_statistics()
        
        print(f"\n{'='*70}")
        print(f"Membership Distribution (Relative Ranking)")
        print(f"{'='*70}")
        print(f"Total Clients: {stats['total_clients']}")
        print(f"\nLevel Distribution:")
        
        for level in reversed(self.levels):  # 从高到低显示
            count = stats['level_distribution'][level]
            percentage = stats['level_percentages'][level]
            avg_rank = stats['average_ranks_by_level'][level]
            
            rank_str = f", Avg Rank: {avg_rank:.1f}" if avg_rank is not None else ""
            print(f"  {level.capitalize():8s}: {count:3d} ({percentage:5.1f}%{rank_str})")
        
        print(f"\nAverage Points: {stats['average_points']:.2f}")
        print(f"{'='*70}")