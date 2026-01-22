"""
incentive/membership.py - 会员等级系统（重构版）
Membership Level System (Refactored)

支持层级约束动态梯度奖励机制的三级会员系统
Three-tier membership system supporting Tier-Constrained Dynamic Gradient Reward

等级划分 / Tier Division:
- Gold (金牌): Top 20% 客户端 / Top 20% clients
- Silver (银牌): Next 30% 客户端 / Next 30% clients  
- Bronze (铜牌): Bottom 50% 客户端 / Bottom 50% clients
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict


class MembershipSystem:
    """
    会员等级管理系统 - 三级制版本
    Membership Level Management System - Three-Tier Version
    
    核心特点 / Core Features:
    1. 使用相对排名而非绝对阈值 / Use relative ranking instead of absolute thresholds
    2. 引入零和博弈特性 / Introduce zero-sum game property
    3. 与层级约束动态梯度奖励配合 / Work with Tier-Constrained Dynamic Gradient Reward
    """
    
    def __init__(self, 
                 ranking_percentiles: Dict[str, float] = None,
                 use_three_tier: bool = True):
        """
        初始化会员系统 / Initialize membership system
        
        Args:
            ranking_percentiles: 等级排名百分位 / Level ranking percentiles
            use_three_tier: 是否使用三级制 / Whether to use three-tier system
        """
        self.use_three_tier = use_three_tier
        
        if use_three_tier:
            # 三级制：Gold, Silver, Bronze
            self.levels = ['bronze', 'silver', 'gold']
            
            if ranking_percentiles is None:
                # 默认配置：Gold 20%, Silver 30%, Bronze 50%
                # Default: Gold 20%, Silver 30%, Bronze 50%
                self.ranking_percentiles = {
                    'gold': 0.80,     # Top 20% -> percentile >= 0.80
                    'silver': 0.50,   # Next 30% -> 0.50 <= percentile < 0.80
                    'bronze': 0.00    # Bottom 50% -> percentile < 0.50
                }
            else:
                self.ranking_percentiles = ranking_percentiles
        else:
            # 四级制：Diamond, Gold, Silver, Bronze（兼容旧版本）
            # Four-tier: Diamond, Gold, Silver, Bronze (backward compatible)
            self.levels = ['bronze', 'silver', 'gold', 'diamond']
            
            if ranking_percentiles is None:
                self.ranking_percentiles = {
                    'diamond': 0.90,  # Top 10%
                    'gold': 0.60,     # Next 30%
                    'silver': 0.20,   # Next 40%
                    'bronze': 0.00    # Bottom 20%
                }
            else:
                self.ranking_percentiles = ranking_percentiles
        
        # 客户端会员信息 / Client membership information
        self.client_memberships = {}
        
        # 历史记录 / History records
        self.level_history = defaultdict(list)
        
        print(f"\n{'='*60}")
        print(f"MembershipSystem Initialized - {'Three-Tier' if use_three_tier else 'Four-Tier'} System")
        print(f"会员系统已初始化 - {'三级制' if use_three_tier else '四级制'}")
        print(f"{'='*60}")
        print(f"Level Distribution / 等级分布:")
        
        # 计算并打印每个等级的比例 / Calculate and print proportion of each level
        sorted_levels = sorted(self.ranking_percentiles.items(), key=lambda x: x[1], reverse=True)
        prev_percentile = 1.0
        for level, percentile in sorted_levels:
            proportion = (prev_percentile - percentile) * 100
            print(f"  {level.capitalize():8s}: Top {int(proportion)}% "
                  f"(percentile >= {percentile:.2f})")
            prev_percentile = percentile
        print(f"{'='*60}\n")
    
    def initialize_client(self, client_id: int) -> None:
        """
        初始化客户端会员信息 / Initialize client membership information
        
        Args:
            client_id: 客户端ID / Client ID
        """
        self.client_memberships[client_id] = {
            'level': 'bronze',
            'total_points': 0.0,
            'active_points': 0.0,
            'ranking_percentile': 0.0,
            'rank': None,
            'contribution_history': [],
            'level_history': [],
            'join_time': datetime.now()
        }
    
    def update_all_memberships_by_ranking(self, 
                                         client_points: Dict[int, float]) -> Dict[int, str]:
        """
        基于相对排名更新所有客户端的会员等级
        Update all memberships by relative ranking
        
        这是核心方法，实现锦标赛选择逻辑
        This is the core method implementing tournament selection logic
        
        Args:
            client_points: 所有客户端的累积信誉分 {client_id: R_{i,t}}
            
        Returns:
            更新后的等级字典 {client_id: new_level}
        """
        if not client_points:
            return {}
        
        # 按累积信誉分降序排列客户端 / Sort clients by accumulated reputation descending
        sorted_clients = sorted(client_points.items(), key=lambda x: x[1], reverse=True)
        total_clients = len(sorted_clients)
        
        new_levels = {}
        tier_counts = defaultdict(int)
        
        for rank, (client_id, points) in enumerate(sorted_clients):
            # 计算排名百分位 (0=最低, 1=最高)
            # Calculate ranking percentile (0=lowest, 1=highest)
            percentile = (total_clients - 1 - rank) / max(1, total_clients - 1)
            
            # 根据百分位确定等级 / Determine level by percentile
            new_level = self._get_level_by_percentile(percentile)
            tier_counts[new_level] += 1
            
            # 更新客户端信息 / Update client info
            if client_id not in self.client_memberships:
                self.initialize_client(client_id)
            
            membership = self.client_memberships[client_id]
            old_level = membership['level']
            
            # 更新信息 / Update information
            membership['level'] = new_level
            membership['total_points'] = points
            membership['active_points'] = points
            membership['ranking_percentile'] = percentile
            membership['rank'] = rank + 1  # 1-indexed rank
            
            new_levels[client_id] = new_level
            
            # 记录等级变化 / Record level change
            if old_level != new_level:
                change_record = {
                    'timestamp': datetime.now(),
                    'old_level': old_level,
                    'new_level': new_level,
                    'points': points,
                    'rank': rank + 1,
                    'percentile': percentile
                }
                membership['level_history'].append(change_record)
                self.level_history[client_id].append(change_record)
        
        return new_levels
    
    def _get_level_by_percentile(self, percentile: float) -> str:
        """
        根据百分位获取等级
        Get level by percentile
        
        Args:
            percentile: 排名百分位 (0-1)
            
        Returns:
            等级名称 / Level name
        """
        # 按百分位阈值从高到低排序 / Sort by percentile threshold descending
        sorted_levels = sorted(self.ranking_percentiles.items(), 
                              key=lambda x: x[1], reverse=True)
        
        for level, threshold in sorted_levels:
            if percentile >= threshold:
                return level
        
        # 默认返回最低等级 / Default to lowest level
        return 'bronze'
    
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
    
    def get_all_client_levels(self) -> Dict[int, str]:
        """
        获取所有客户端的等级
        Get levels for all clients
        
        Returns:
            等级字典 {client_id: level}
        """
        return {cid: info['level'] for cid, info in self.client_memberships.items()}
    
    def get_clients_by_level(self, level: str) -> List[int]:
        """
        获取指定等级的所有客户端
        Get all clients of a specified level
        
        Args:
            level: 等级名称 / Level name
            
        Returns:
            客户端ID列表 / List of client IDs
        """
        return [cid for cid, info in self.client_memberships.items() 
                if info['level'] == level]
    
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
        total_points = 0.0
        ranks_by_level = {level: [] for level in self.levels}
        percentiles_by_level = {level: [] for level in self.levels}
        
        for client_id, membership in self.client_memberships.items():
            level = membership['level']
            level_counts[level] = level_counts.get(level, 0) + 1
            total_points += membership['total_points']
            
            if membership['rank'] is not None:
                ranks_by_level[level].append(membership['rank'])
            percentiles_by_level[level].append(membership['ranking_percentile'])
        
        num_clients = len(self.client_memberships)
        avg_points = total_points / num_clients if num_clients > 0 else 0
        
        # 计算每个等级的统计 / Calculate statistics per level
        level_stats = {}
        for level in self.levels:
            if ranks_by_level[level]:
                level_stats[level] = {
                    'count': level_counts.get(level, 0),
                    'avg_rank': np.mean(ranks_by_level[level]),
                    'avg_percentile': np.mean(percentiles_by_level[level]),
                    'min_rank': min(ranks_by_level[level]) if ranks_by_level[level] else None,
                    'max_rank': max(ranks_by_level[level]) if ranks_by_level[level] else None
                }
            else:
                level_stats[level] = {
                    'count': level_counts.get(level, 0),
                    'avg_rank': None,
                    'avg_percentile': None,
                    'min_rank': None,
                    'max_rank': None
                }
        
        return {
            'total_clients': num_clients,
            'level_distribution': level_counts,
            'level_percentages': {
                level: count / num_clients * 100 if num_clients > 0 else 0
                for level, count in level_counts.items()
            },
            'total_points': total_points,
            'average_points': avg_points,
            'level_stats': level_stats
        }
    
    def update_contribution_history(self, 
                                   client_id: int, 
                                   contribution: float,
                                   round_num: int) -> None:
        """
        更新客户端的贡献历史
        Update client's contribution history
        
        Args:
            client_id: 客户端ID / Client ID
            contribution: 贡献度 / Contribution
            round_num: 轮次 / Round number
        """
        if client_id not in self.client_memberships:
            self.initialize_client(client_id)
        
        self.client_memberships[client_id]['contribution_history'].append({
            'round': round_num,
            'contribution': contribution,
            'timestamp': datetime.now()
        })
    
    def print_membership_distribution(self) -> None:
        """打印会员等级分布 / Print membership distribution"""
        stats = self.get_membership_statistics()
        
        print(f"\n{'='*70}")
        print(f"Membership Distribution (Relative Ranking)")
        print(f"会员等级分布（相对排名）")
        print(f"{'='*70}")
        print(f"Total Clients / 客户端总数: {stats['total_clients']}")
        print(f"\nLevel Distribution / 等级分布:")
        
        # 期望比例 / Expected ratios
        if self.use_three_tier:
            expected = {'gold': 20, 'silver': 30, 'bronze': 50}
        else:
            expected = {'diamond': 10, 'gold': 30, 'silver': 40, 'bronze': 20}
        
        for level in reversed(self.levels):
            count = stats['level_distribution'].get(level, 0)
            percentage = stats['level_percentages'].get(level, 0)
            level_info = stats['level_stats'].get(level, {})
            
            avg_rank = level_info.get('avg_rank')
            rank_str = f", Avg Rank: {avg_rank:.1f}" if avg_rank is not None else ""
            expected_str = f" (Expected: {expected.get(level, 0)}%)"
            
            print(f"  {level.capitalize():8s}: {count:3d} ({percentage:5.1f}%{expected_str}{rank_str})")
        
        print(f"\nAverage Points / 平均积分: {stats['average_points']:.2f}")
        print(f"{'='*70}")
    
    def get_level_transition_summary(self) -> Dict:
        """
        获取等级变化摘要
        Get level transition summary
        
        Returns:
            等级变化统计 / Level transition statistics
        """
        transitions = defaultdict(int)
        
        for client_id, history in self.level_history.items():
            for change in history:
                key = f"{change['old_level']} -> {change['new_level']}"
                transitions[key] += 1
        
        return dict(transitions)
    
    def reset(self) -> None:
        """重置会员系统 / Reset membership system"""
        self.client_memberships.clear()
        self.level_history.clear()