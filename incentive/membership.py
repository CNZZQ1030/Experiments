"""
incentive/membership.py (Updated for PDF Specification)
会员等级系统 - 基于相对排名 / Membership Level System - Based on Relative Ranking

按照PDF文档定义的比例 / According to PDF specification:
- Diamond (钻石): Top 10%
- Gold (金牌): Next 30%
- Silver (银牌): Next 40%
- Bronze (铜牌): Bottom 20%
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


class MembershipSystem:
    """
    会员等级管理系统 - 基于相对排名 / Membership Level Management System - Relative Ranking
    
    核心特点 / Core Features:
    1. 使用相对排名而非绝对阈值 / Use relative ranking instead of absolute thresholds
    2. 引入零和博弈特性 / Introduce zero-sum game property
    3. 按照PDF规定的比例分配等级 / Allocate levels according to PDF specification
    """
    
    def __init__(self, level_multipliers: Dict[str, float] = None,
                 ranking_percentiles: Dict[str, float] = None):
        """
        初始化会员系统 / Initialize membership system
        
        Args:
            level_multipliers: 等级倍数 / Level multipliers
            ranking_percentiles: 等级排名百分位 / Level ranking percentiles
        """
        self.levels = ['bronze', 'silver', 'gold', 'diamond']
        
        # 默认等级倍数 / Default level multipliers
        if level_multipliers is None:
            self.level_multipliers = {
                'bronze': 1.0,
                'silver': 1.2,
                'gold': 1.5,
                'diamond': 2.0
            }
        else:
            self.level_multipliers = level_multipliers
        
        # 使用PDF定义的百分位：Diamond 10%, Gold 30%, Silver 40%, Bronze 20%
        # Using PDF-defined percentiles
        # percentile值表示该等级需要的最低排名百分位
        # The percentile value indicates the minimum ranking percentile for that level
        if ranking_percentiles is None:
            self.ranking_percentiles = {
                'diamond': 0.90,  # Top 10% -> percentile >= 0.90
                'gold': 0.60,     # Next 30% -> 0.60 <= percentile < 0.90
                'silver': 0.20,   # Next 40% -> 0.20 <= percentile < 0.60
                'bronze': 0.00    # Bottom 20% -> percentile < 0.20
            }
        else:
            self.ranking_percentiles = ranking_percentiles
        
        # 客户端会员信息 / Client membership information
        self.client_memberships = {}
        
        print(f"MembershipSystem initialized with RELATIVE RANKING:")
        print(f"  Diamond: Top {(1-self.ranking_percentiles['diamond'])*100:.0f}%")
        print(f"  Gold: Next {(self.ranking_percentiles['diamond']-self.ranking_percentiles['gold'])*100:.0f}%")
        print(f"  Silver: Next {(self.ranking_percentiles['gold']-self.ranking_percentiles['silver'])*100:.0f}%")
        print(f"  Bronze: Bottom {self.ranking_percentiles['silver']*100:.0f}%")
        
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
            'ranking_percentile': 0.0,
            'rank': None,
            'level_history': [],
            'join_time': datetime.now()
        }
    
    def update_all_memberships_by_ranking(self, client_points: Dict[int, float]) -> Dict[int, str]:
        """
        基于相对排名更新所有客户端的会员等级 / Update all memberships by relative ranking
        
        这是核心方法！实现PDF文档中描述的锦标赛选择逻辑
        This is the core method! Implements the tournament selection logic from PDF
        
        Args:
            client_points: 所有客户端的累积信誉分 {client_id: R_{i,t}}
            
        Returns:
            更新后的等级字典 {client_id: new_level}
        """
        if not client_points:
            return {}
        
        # 按累积信誉分降序排列客户端 / Sort clients by accumulated reputation score descending
        sorted_clients = sorted(client_points.items(), key=lambda x: x[1], reverse=True)
        total_clients = len(sorted_clients)
        
        new_levels = {}
        
        for rank, (client_id, points) in enumerate(sorted_clients, start=1):
            # 计算排名百分位 (0=最低, 1=最高)
            # Calculate ranking percentile (0=lowest, 1=highest)
            # percentile = (total_clients - rank) / total_clients
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
        
        return new_levels
    
    def get_client_level(self, client_id: int) -> str:
        """
        获取客户端等级 / Get client level
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            等级名称 / Level name
        """
        if client_id not in self.client_memberships:
            return 'bronze'
        return self.client_memberships[client_id]['level']
    
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
            'level_index': self.levels.index(level) if level in self.levels else 0
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
        
        ranks_by_level = {level: [] for level in self.levels}
        
        for membership in self.client_memberships.values():
            level_counts[membership['level']] += 1
            total_points += membership['total_points']
            
            if membership['rank'] is not None:
                ranks_by_level[membership['level']].append(membership['rank'])
        
        num_clients = len(self.client_memberships)
        avg_points = total_points / num_clients if num_clients > 0 else 0
        
        # 计算每个等级的平均排名 / Calculate average rank per level
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
        
        # 期望比例 / Expected ratios
        expected = {'diamond': 10, 'gold': 30, 'silver': 40, 'bronze': 20}
        
        for level in reversed(self.levels):
            count = stats['level_distribution'][level]
            percentage = stats['level_percentages'][level]
            avg_rank = stats['average_ranks_by_level'][level]
            
            rank_str = f", Avg Rank: {avg_rank:.1f}" if avg_rank is not None else ""
            expected_str = f" (Expected: {expected[level]}%)"
            print(f"  {level.capitalize():8s}: {count:3d} ({percentage:5.1f}%{expected_str}{rank_str})")
        
        print(f"\nAverage Points: {stats['average_points']:.2f}")
        print(f"{'='*70}")