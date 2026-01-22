"""
incentive/time_slice.py - 时间片管理器
Time Slice Manager

实时积分累加，阶段性失效
Real-time point accumulation, periodic expiration
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
    """
    
    def __init__(self, 
                 slice_type: str = "rounds", 
                 rounds_per_slice: int = 5,
                 days_per_slice: int = 3,
                 validity_slices: int = 2):
        """
        初始化时间片管理器 / Initialize time slice manager
        
        Args:
            slice_type: 时间片类型 / Time slice type
            rounds_per_slice: 每个时间片的轮次数 / Rounds per slice
            days_per_slice: 每个时间片的天数 / Days per slice
            validity_slices: 积分有效期（时间片数）/ Points validity period
        """
        self.slice_type = slice_type
        self.rounds_per_slice = rounds_per_slice
        self.days_per_slice = days_per_slice
        self.validity_slices = validity_slices
        
        self.current_slice = 0
        self.slice_start_time = datetime.now()
        
        # 客户端时间片积分存储 / Client slice points storage
        self.client_slice_points = {}
        
        # 统计信息 / Statistics
        self.total_points_awarded = 0
        self.total_points_expired = 0
        
        print(f"TimeSliceManager initialized:")
        print(f"  Type: {slice_type}")
        print(f"  Rounds per slice: {rounds_per_slice}")
        print(f"  Validity: {validity_slices} slices")
    
    def get_current_slice(self, round_num: int) -> int:
        """获取当前时间片编号 / Get current time slice number"""
        if self.slice_type == "rounds":
            return round_num // self.rounds_per_slice
        elif self.slice_type == "days":
            days_elapsed = (datetime.now() - self.slice_start_time).days
            return days_elapsed // self.days_per_slice
        else:
            return round_num // self.rounds_per_slice
    
    def add_contribution_points(self, client_id: int, round_num: int, 
                               contribution: float) -> float:
        """
        实时添加贡献积分到当前时间片
        Add contribution points to current slice in real-time
        
        Args:
            client_id: 客户端ID / Client ID
            round_num: 当前轮次 / Current round
            contribution: 贡献度 (0-1) / Contribution
            
        Returns:
            当前有效总积分 / Current total active points
        """
        points = contribution * 1000
        current_slice = self.get_current_slice(round_num)
        
        if client_id not in self.client_slice_points:
            self.client_slice_points[client_id] = {}
        
        if current_slice not in self.client_slice_points[client_id]:
            self.client_slice_points[client_id][current_slice] = 0
        
        self.client_slice_points[client_id][current_slice] += points
        self.total_points_awarded += points
        
        return self.get_active_points(client_id, round_num)
    
    def get_active_points(self, client_id: int, current_round: int) -> float:
        """
        获取有效期内的积分总和
        Get sum of points within validity period
        """
        if client_id not in self.client_slice_points:
            return 0.0
        
        current_slice = self.get_current_slice(current_round)
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        active_points = 0.0
        for slice_num, points in self.client_slice_points[client_id].items():
            if min_valid_slice <= slice_num <= current_slice:
                active_points += points
        
        return active_points
    
    def clean_expired_points(self, current_round: int) -> Dict[int, float]:
        """
        清理过期积分 / Clean expired points
        """
        current_slice = self.get_current_slice(current_round)
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        cleaned_points = {}
        
        for client_id in self.client_slice_points:
            client_cleaned = 0.0
            expired_slices = []
            
            for slice_num in self.client_slice_points[client_id]:
                if slice_num < min_valid_slice:
                    client_cleaned += self.client_slice_points[client_id][slice_num]
                    expired_slices.append(slice_num)
            
            for slice_num in expired_slices:
                del self.client_slice_points[client_id][slice_num]
            
            if client_cleaned > 0:
                cleaned_points[client_id] = client_cleaned
                self.total_points_expired += client_cleaned
        
        return cleaned_points
    
    def get_all_client_active_points(self, current_round: int) -> Dict[int, float]:
        """获取所有客户端的当前有效积分 / Get active points for all clients"""
        result = {}
        for client_id in self.client_slice_points.keys():
            result[client_id] = self.get_active_points(client_id, current_round)
        return result
    
    def get_statistics(self) -> Dict:
        """获取统计信息 / Get statistics"""
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
            'active_points_std': np.std(all_active_points) if all_active_points else 0
        }
    
    def print_summary(self, current_round: int) -> None:
        """打印当前状态摘要 / Print current state summary"""
        current_slice = self.get_current_slice(current_round)
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"Time Slice Summary - Round {current_round}")
        print(f"{'='*70}")
        print(f"Current Slice: {current_slice}")
        print(f"Total Points Awarded: {stats['total_points_awarded']:.2f}")
        print(f"Total Points Expired: {stats['total_points_expired']:.2f}")
        print(f"Active Points Mean: {stats['active_points_mean']:.2f}")
        print(f"{'='*70}")