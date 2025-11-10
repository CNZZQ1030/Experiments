"""
incentive/time_slice.py
时间片管理器 / Time Slice Manager
"""

from datetime import datetime

class TimeSliceManager:
    """
    时间片管理器 / Time Slice Manager
    管理不同类型的时间片策略
    Manage different types of time slice strategies
    """
    
    def __init__(self, slice_type: str = "rounds", 
                 rounds_per_slice: int = 10,
                 days_per_slice: int = 3,
                 validity_slices: int = 10):
        """
        初始化时间片管理器 / Initialize time slice manager
        
        Args:
            slice_type: 时间片类型 / Time slice type
            rounds_per_slice: 每个时间片的轮次数 / Rounds per time slice
            days_per_slice: 每个时间片的天数 / Days per time slice
            validity_slices: 积分有效期(时间片数) / Points validity (number of slices)
        """
        self.slice_type = slice_type
        self.rounds_per_slice = rounds_per_slice
        self.days_per_slice = days_per_slice
        self.validity_slices = validity_slices
        
        # 时间片历史 / Time slice history
        self.current_slice = 0
        self.slice_start_time = datetime.now()
        self.slice_history = []
        
        # 客户端时间片积分 / Client time slice points
        self.client_slice_points = {}
        
    def get_current_slice(self, round_num: int) -> int:
        """
        获取当前时间片 / Get current time slice
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            时间片编号 / Time slice number
        """
        if self.slice_type == "rounds":
            return self._get_rounds_based_slice(round_num)
        elif self.slice_type == "days":
            return self._get_days_based_slice()
        elif self.slice_type == "phases":
            return self._get_phase_based_slice(round_num)
        elif self.slice_type == "dynamic":
            return self._get_dynamic_slice(round_num)
        else:
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
        # 假设训练分为4个阶段 / Assume training is divided into 4 phases
        total_rounds = 100  # 应从配置获取 / Should get from config
        phase_length = total_rounds // 4
        return round_num // phase_length
    
    def _get_dynamic_slice(self, round_num: int) -> int:
        """动态时间片 / Dynamic time slice"""
        # 根据系统活跃度动态调整 / Dynamically adjust based on system activity
        if hasattr(self, 'system_activity') and self.system_activity < 0.5:
            # 活跃度低时缩短时间片 / Shorten slice when activity is low
            return round_num // (self.rounds_per_slice // 2)
        return round_num // self.rounds_per_slice
    
    def _get_completion_based_slice(self, round_num: int) -> int:
        """基于任务完成度的时间片 / Completion-based time slice"""
        # 每完成25%为一个时间片 / Each 25% completion is a slice
        total_rounds = 100  # 应从配置获取 / Should get from config
        completion_rate = round_num / total_rounds
        return int(completion_rate * 4)
    
    def update_client_slice_points(self, client_id: int, round_num: int, 
                                  points: float) -> None:
        """
        更新客户端时间片积分 / Update client slice points
        
        Args:
            client_id: 客户端ID / Client ID
            round_num: 轮次 / Round number
            points: 积分 / Points
        """
        current_slice = self.get_current_slice(round_num)
        
        if client_id not in self.client_slice_points:
            self.client_slice_points[client_id] = {}
        
        if current_slice not in self.client_slice_points[client_id]:
            self.client_slice_points[client_id][current_slice] = 0
        
        self.client_slice_points[client_id][current_slice] += points
    
    def get_active_points(self, client_id: int, current_round: int) -> float:
        """
        获取有效期内的积分 / Get points within validity period
        
        Args:
            client_id: 客户端ID / Client ID
            current_round: 当前轮次 / Current round
            
        Returns:
            有效积分 / Valid points
        """
        if client_id not in self.client_slice_points:
            return 0
        
        current_slice = self.get_current_slice(current_round)
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        active_points = 0
        for slice_num, points in self.client_slice_points[client_id].items():
            if min_valid_slice <= slice_num <= current_slice:
                active_points += points
        
        return active_points
    
    def clean_expired_points(self, current_round: int) -> None:
        """
        清理过期积分 / Clean expired points
        
        Args:
            current_round: 当前轮次 / Current round
        """
        current_slice = self.get_current_slice(current_round)
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        for client_id in self.client_slice_points:
            expired_slices = []
            for slice_num in self.client_slice_points[client_id]:
                if slice_num < min_valid_slice:
                    expired_slices.append(slice_num)
            
            for slice_num in expired_slices:
                del self.client_slice_points[client_id][slice_num]
    
    def set_system_activity(self, activity: float) -> None:
        """
        设置系统活跃度（用于动态时间片） / Set system activity (for dynamic slicing)
        
        Args:
            activity: 系统活跃度 / System activity
        """
        self.system_activity = activity