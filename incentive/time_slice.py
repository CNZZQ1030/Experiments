"""
incentive/time_slice.py (Refactored)
时间片管理器 - 实时积分累加，阶段性失效
Time Slice Manager - Real-time point accumulation, periodic expiration
"""

from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple


class TimeSliceManager:
    """
    时间片管理器 / Time Slice Manager
    
    核心思路 / Core Concept:
    - 积分获取是实时的：每轮训练后立即累加到当前时间片
    - 积分失效是阶段性的：当时间片结束时，最旧的时间片积分失效
    
    Real-time point accumulation: Points added to current slice immediately after each round
    Periodic expiration: Oldest slice points expire when time slice ends
    """
    
    def __init__(self, 
                 slice_type: str = "rounds", 
                 rounds_per_slice: int = 10,
                 days_per_slice: int = 3,
                 validity_slices: int = 10):
        """
        初始化时间片管理器 / Initialize time slice manager
        
        Args:
            slice_type: 时间片类型 / Time slice type ("rounds", "days", "phases", "dynamic", "completion")
            rounds_per_slice: 每个时间片的轮次数 / Rounds per slice
            days_per_slice: 每个时间片的天数 / Days per slice
            validity_slices: 积分有效期（时间片数）/ Points validity period (number of slices)
        """
        self.slice_type = slice_type
        self.rounds_per_slice = rounds_per_slice
        self.days_per_slice = days_per_slice
        self.validity_slices = validity_slices
        
        # 时间片历史 / Slice history
        self.current_slice = 0
        self.slice_start_time = datetime.now()
        
        # 客户端时间片积分存储 / Client slice points storage
        # {client_id: {slice_num: points}}
        self.client_slice_points = {}
        
        # 统计信息 / Statistics
        self.total_points_awarded = 0
        self.total_points_expired = 0
        
        print(f"TimeSliceManager initialized:")
        print(f"  Type: {slice_type}")
        print(f"  Rounds per slice: {rounds_per_slice}")
        print(f"  Validity: {validity_slices} slices")
        print(f"  Strategy: Real-time accumulation + Periodic expiration")
    
    # =========================================================================
    # 时间片计算 / Time Slice Calculation
    # =========================================================================
    
    def get_current_slice(self, round_num: int) -> int:
        """
        获取当前时间片编号 / Get current time slice number
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            时间片编号 / Slice number
        """
        if self.slice_type == "rounds":
            return self._get_rounds_based_slice(round_num)
        elif self.slice_type == "days":
            return self._get_days_based_slice()
        elif self.slice_type == "phases":
            return self._get_phase_based_slice(round_num)
        elif self.slice_type == "dynamic":
            return self._get_dynamic_slice(round_num)
        else:  # completion
            return self._get_completion_based_slice(round_num)
    
    def _get_rounds_based_slice(self, round_num: int) -> int:
        """基于轮次的时间片 / Rounds-based time slice"""
        return round_num // self.rounds_per_slice
    
    def _get_days_based_slice(self) -> int:
        """基于天数的时间片 / Days-based time slice"""
        days_elapsed = (datetime.now() - self.slice_start_time).days
        return days_elapsed // self.days_per_slice
    
    def _get_phase_based_slice(self, round_num: int) -> int:
        """基于训练阶段的时间片 / Phase-based time slice"""
        total_rounds = 100
        phase_length = total_rounds // 4
        return round_num // phase_length
    
    def _get_dynamic_slice(self, round_num: int) -> int:
        """动态时间片 / Dynamic time slice"""
        if hasattr(self, 'system_activity') and self.system_activity < 0.5:
            return round_num // (self.rounds_per_slice // 2)
        return round_num // self.rounds_per_slice
    
    def _get_completion_based_slice(self, round_num: int) -> int:
        """基于任务完成度的时间片 / Completion-based time slice"""
        total_rounds = 100
        completion_rate = round_num / total_rounds
        return int(completion_rate * 4)
    
    # =========================================================================
    # 核心功能：实时积分累加 / Core: Real-time Point Accumulation
    # =========================================================================
    
    def add_contribution_points(self, client_id: int, round_num: int, 
                               contribution: float) -> float:
        """
        实时添加贡献积分到当前时间片 / Add contribution points to current slice in real-time
        
        每轮训练后立即调用，将贡献度转换为积分并累加到当前时间片
        Called immediately after each round, converts contribution to points and adds to current slice
        
        Args:
            client_id: 客户端ID / Client ID
            round_num: 当前轮次 / Current round
            contribution: 贡献度 (0-1) / Contribution (0-1)
            
        Returns:
            当前有效总积分 / Current total active points
        """
        # 转换为积分 / Convert to points
        points = contribution * 1000  # 贡献度 × 1000 转为积分
        
        # 获取当前时间片 / Get current slice
        current_slice = self.get_current_slice(round_num)
        
        # 初始化客户端数据 / Initialize client data
        if client_id not in self.client_slice_points:
            self.client_slice_points[client_id] = {}
        
        # 立即累加到当前时间片 / Immediately add to current slice
        if current_slice not in self.client_slice_points[client_id]:
            self.client_slice_points[client_id][current_slice] = 0
        
        self.client_slice_points[client_id][current_slice] += points
        self.total_points_awarded += points
        
        # 返回当前有效总积分 / Return current total active points
        return self.get_active_points(client_id, round_num)
    
    # =========================================================================
    # 核心功能：阶段性积分失效 / Core: Periodic Point Expiration
    # =========================================================================
    
    def get_active_points(self, client_id: int, current_round: int) -> float:
        """
        获取有效期内的积分总和 / Get sum of points within validity period
        
        只统计最近N个时间片的积分，过期的时间片不计入
        Only counts points from recent N slices, expired slices excluded
        
        Args:
            client_id: 客户端ID / Client ID
            current_round: 当前轮次 / Current round
            
        Returns:
            有效积分总和 / Sum of active points
        """
        if client_id not in self.client_slice_points:
            return 0.0
        
        current_slice = self.get_current_slice(current_round)
        # 计算有效时间片的起始编号 / Calculate start of valid slices
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        # 累加有效期内的所有积分 / Sum all points within validity period
        active_points = 0.0
        for slice_num, points in self.client_slice_points[client_id].items():
            if min_valid_slice <= slice_num <= current_slice:
                active_points += points
        
        return active_points
    
    def clean_expired_points(self, current_round: int) -> Dict[int, float]:
        """
        清理过期积分 / Clean expired points
        
        当时间片结束时调用，移除超出有效期的旧时间片积分
        Called when time slice ends, removes points from slices beyond validity period
        
        Args:
            current_round: 当前轮次 / Current round
            
        Returns:
            每个客户端清理的积分数量 / Amount of points cleaned per client
        """
        current_slice = self.get_current_slice(current_round)
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        cleaned_points = {}
        
        for client_id in self.client_slice_points:
            client_cleaned = 0.0
            expired_slices = []
            
            # 找出过期的时间片 / Find expired slices
            for slice_num in self.client_slice_points[client_id]:
                if slice_num < min_valid_slice:
                    client_cleaned += self.client_slice_points[client_id][slice_num]
                    expired_slices.append(slice_num)
            
            # 删除过期时间片 / Delete expired slices
            for slice_num in expired_slices:
                del self.client_slice_points[client_id][slice_num]
            
            if client_cleaned > 0:
                cleaned_points[client_id] = client_cleaned
                self.total_points_expired += client_cleaned
        
        if cleaned_points:
            print(f"Round {current_round}: Cleaned expired points from {len(cleaned_points)} clients")
        
        return cleaned_points
    
    # =========================================================================
    # 辅助功能 / Helper Functions
    # =========================================================================
    
    def get_client_slice_breakdown(self, client_id: int, current_round: int) -> Dict:
        """
        获取客户端的时间片积分明细 / Get client's slice points breakdown
        
        Args:
            client_id: 客户端ID / Client ID
            current_round: 当前轮次 / Current round
            
        Returns:
            时间片积分明细 / Slice points breakdown
        """
        if client_id not in self.client_slice_points:
            return {}
        
        current_slice = self.get_current_slice(current_round)
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        breakdown = {
            'current_slice': current_slice,
            'valid_slice_range': (min_valid_slice, current_slice),
            'slice_points': {},
            'total_active_points': 0.0
        }
        
        for slice_num, points in sorted(self.client_slice_points[client_id].items()):
            status = 'active' if min_valid_slice <= slice_num <= current_slice else 'expired'
            breakdown['slice_points'][slice_num] = {
                'points': points,
                'status': status
            }
            if status == 'active':
                breakdown['total_active_points'] += points
        
        return breakdown
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息 / Get statistics
        
        Returns:
            统计信息 / Statistics
        """
        all_active_points = []
        slices_per_client = []
        
        for client_id, slice_points in self.client_slice_points.items():
            all_active_points.append(sum(slice_points.values()))
            slices_per_client.append(len(slice_points))
        
        return {
            'slice_type': self.slice_type,
            'rounds_per_slice': self.rounds_per_slice,
            'validity_slices': self.validity_slices,
            'total_clients': len(self.client_slice_points),
            'total_points_awarded': self.total_points_awarded,
            'total_points_expired': self.total_points_expired,
            'active_points_mean': np.mean(all_active_points) if all_active_points else 0,
            'active_points_std': np.std(all_active_points) if all_active_points else 0,
            'avg_slices_per_client': np.mean(slices_per_client) if slices_per_client else 0
        }
    
    def get_all_client_active_points(self, current_round: int) -> Dict[int, float]:
        """
        获取所有客户端的当前有效积分 / Get current active points for all clients
        
        Args:
            current_round: 当前轮次 / Current round
            
        Returns:
            客户端ID到有效积分的映射 / Mapping from client ID to active points
        """
        result = {}
        for client_id in self.client_slice_points.keys():
            result[client_id] = self.get_active_points(client_id, current_round)
        return result
    
    def set_system_activity(self, activity: float) -> None:
        """设置系统活跃度（用于动态时间片）/ Set system activity (for dynamic slicing)"""
        self.system_activity = activity
    
    def print_summary(self, current_round: int) -> None:
        """
        打印当前状态摘要 / Print current state summary
        
        Args:
            current_round: 当前轮次 / Current round
        """
        current_slice = self.get_current_slice(current_round)
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"Time Slice Summary - Round {current_round}")
        print(f"{'='*70}")
        print(f"Current Slice: {current_slice}")
        print(f"Total Points Awarded: {stats['total_points_awarded']:.2f}")
        print(f"Total Points Expired: {stats['total_points_expired']:.2f}")
        print(f"Active Points Mean: {stats['active_points_mean']:.2f}")
        print(f"Active Points Std: {stats['active_points_std']:.2f}")
        print(f"{'='*70}")