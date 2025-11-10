"""
incentive/time_slice.py (EMA Enhanced Version)
时间片管理器 - 支持EMA和滑动窗口双模式
Time Slice Manager - Supports both EMA and Sliding Window modes
"""

from datetime import datetime
import numpy as np
from typing import Dict, Optional, List, Tuple


class TimeSliceManager:
    """
    时间片管理器 / Time Slice Manager
    支持三种模式：EMA、滑动窗口、混合模式
    Supports three modes: EMA, Sliding Window, Hybrid
    """
    
    def __init__(self, 
                 smoothing_method: str = "ema",
                 ema_alpha: float = 0.3,
                 use_warmup: bool = True,
                 warmup_rounds: int = 20,
                 warmup_boost: float = 1.5,
                 slice_type: str = "rounds", 
                 rounds_per_slice: int = 10,
                 days_per_slice: int = 3,
                 validity_slices: int = 10,
                 hybrid_ema_weight: float = 0.7,
                 hybrid_window_size: int = 5):
        """
        初始化时间片管理器 / Initialize time slice manager
        
        Args:
            smoothing_method: 平滑方法 ("ema", "sliding_window", "hybrid")
            ema_alpha: EMA平滑系数 / EMA smoothing factor
            use_warmup: 是否使用预热加速 / Whether to use warm-up acceleration
            warmup_rounds: 预热轮次数 / Number of warm-up rounds
            warmup_boost: 预热期贡献度提升倍数 / Contribution boost during warm-up
            slice_type: 时间片类型 / Time slice type (for sliding_window mode)
            rounds_per_slice: 每个时间片的轮次数 / Rounds per slice
            days_per_slice: 每个时间片的天数 / Days per slice
            validity_slices: 积分有效期 / Points validity period
            hybrid_ema_weight: 混合模式EMA权重 / EMA weight in hybrid mode
            hybrid_window_size: 混合模式窗口大小 / Window size in hybrid mode
        """
        self.smoothing_method = smoothing_method
        self.ema_alpha = ema_alpha
        self.use_warmup = use_warmup
        self.warmup_rounds = warmup_rounds
        self.warmup_boost = warmup_boost
        
        # 滑动窗口参数
        self.slice_type = slice_type
        self.rounds_per_slice = rounds_per_slice
        self.days_per_slice = days_per_slice
        self.validity_slices = validity_slices
        
        # 混合模式参数
        self.hybrid_ema_weight = hybrid_ema_weight
        self.hybrid_window_weight = 1.0 - hybrid_ema_weight
        self.hybrid_window_size = hybrid_window_size
        
        # 时间片历史（用于滑动窗口模式）
        self.current_slice = 0
        self.slice_start_time = datetime.now()
        self.slice_history = []
        
        # === EMA模式数据存储 ===
        self.client_ema = {}  # {client_id: ema_value}
        self.client_participation_count = {}  # 客户端参与次数
        
        # === 滑动窗口模式数据存储 ===
        self.client_slice_points = {}  # {client_id: {slice_num: points}}
        
        # === 混合模式数据存储 ===
        self.client_recent_window = {}  # {client_id: [recent_contributions]}
        
        # === 统计信息 ===
        self.contribution_history = []  # 用于分析
        
        print(f"TimeSliceManager initialized with method: {smoothing_method}")
        if smoothing_method == "ema":
            equiv_window = 2 / ema_alpha - 1
            print(f"EMA α={ema_alpha}, equivalent window≈{equiv_window:.1f} rounds")
        elif smoothing_method == "sliding_window":
            print(f"Sliding Window: validity={validity_slices} slices")
        else:
            print(f"Hybrid: EMA_weight={hybrid_ema_weight}, window_size={hybrid_window_size}")
    
    # =========================================================================
    # EMA模式方法 / EMA Mode Methods
    # =========================================================================
    
    def update_ema(self, client_id: int, contribution: float, round_num: int) -> float:
        """
        使用EMA更新客户端贡献度 / Update client contribution using EMA
        
        Args:
            client_id: 客户端ID / Client ID
            contribution: 当前轮次贡献度 / Current round contribution
            round_num: 当前轮次 / Current round
            
        Returns:
            更新后的EMA值 / Updated EMA value
        """
        # 预热期加速
        if self.use_warmup and round_num <= self.warmup_rounds:
            contribution *= self.warmup_boost
        
        if client_id not in self.client_ema:
            # 第一次参与，EMA初始化为当前贡献度
            self.client_ema[client_id] = contribution
            self.client_participation_count[client_id] = 1
        else:
            # EMA公式: EMA_new = α × new_value + (1-α) × EMA_old
            old_ema = self.client_ema[client_id]
            self.client_ema[client_id] = (
                self.ema_alpha * contribution + 
                (1 - self.ema_alpha) * old_ema
            )
            self.client_participation_count[client_id] += 1
        
        return self.client_ema[client_id]
    
    def get_ema(self, client_id: int) -> float:
        """
        获取客户端的EMA值 / Get client's EMA value
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            EMA值 / EMA value
        """
        return self.client_ema.get(client_id, 0.0)
    
    def get_ema_with_confidence(self, client_id: int) -> Tuple[float, float]:
        """
        获取EMA值及其置信度 / Get EMA value with confidence
        置信度基于参与次数：参与越多，置信度越高
        Confidence based on participation count: more participation, higher confidence
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            (EMA值, 置信度) / (EMA value, confidence)
        """
        ema = self.get_ema(client_id)
        participation = self.client_participation_count.get(client_id, 0)
        
        # 置信度计算：参与次数越多越接近1
        # Confidence calculation: approaches 1 as participation increases
        confidence = min(1.0, participation / 10.0)  # 10次后达到满置信度
        
        return ema, confidence
    
    # =========================================================================
    # 滑动窗口模式方法 / Sliding Window Mode Methods
    # =========================================================================
    
    def get_current_slice(self, round_num: int) -> int:
        """获取当前时间片 / Get current time slice"""
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
    
    def update_client_slice_points(self, client_id: int, round_num: int, 
                                   points: float) -> None:
        """更新客户端时间片积分（滑动窗口模式）"""
        current_slice = self.get_current_slice(round_num)
        
        if client_id not in self.client_slice_points:
            self.client_slice_points[client_id] = {}
        
        if current_slice not in self.client_slice_points[client_id]:
            self.client_slice_points[client_id][current_slice] = 0
        
        self.client_slice_points[client_id][current_slice] += points
    
    def get_active_points(self, client_id: int, current_round: int) -> float:
        """
        获取有效期内的积分（滑动窗口模式）
        Get points within validity period (sliding window mode)
        """
        if client_id not in self.client_slice_points:
            return 0.0
        
        current_slice = self.get_current_slice(current_round)
        # 使用max(0, ...)防止前期时间片不足的问题
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        active_points = 0.0
        for slice_num, points in self.client_slice_points[client_id].items():
            if min_valid_slice <= slice_num <= current_slice:
                active_points += points
        
        return active_points
    
    def clean_expired_points(self, current_round: int) -> None:
        """清理过期积分（滑动窗口模式）"""
        current_slice = self.get_current_slice(current_round)
        min_valid_slice = max(0, current_slice - self.validity_slices + 1)
        
        for client_id in self.client_slice_points:
            expired_slices = []
            for slice_num in self.client_slice_points[client_id]:
                if slice_num < min_valid_slice:
                    expired_slices.append(slice_num)
            
            for slice_num in expired_slices:
                del self.client_slice_points[client_id][slice_num]
    
    # =========================================================================
    # 混合模式方法 / Hybrid Mode Methods
    # =========================================================================
    
    def update_hybrid(self, client_id: int, contribution: float, round_num: int) -> float:
        """
        使用混合模式更新贡献度 / Update contribution using hybrid mode
        结合EMA的平滑性和滑动窗口的精确性
        Combines smoothness of EMA and precision of sliding window
        
        Args:
            client_id: 客户端ID / Client ID
            contribution: 当前贡献度 / Current contribution
            round_num: 当前轮次 / Current round
            
        Returns:
            混合贡献度值 / Hybrid contribution value
        """
        # 1. 更新EMA部分
        ema_value = self.update_ema(client_id, contribution, round_num)
        
        # 2. 更新小窗口部分
        if client_id not in self.client_recent_window:
            self.client_recent_window[client_id] = []
        
        self.client_recent_window[client_id].append(contribution)
        
        # 保持窗口大小
        if len(self.client_recent_window[client_id]) > self.hybrid_window_size:
            self.client_recent_window[client_id].pop(0)
        
        # 3. 计算窗口平均值
        window_avg = np.mean(self.client_recent_window[client_id])
        
        # 4. 加权组合
        hybrid_value = (
            self.hybrid_ema_weight * ema_value + 
            self.hybrid_window_weight * window_avg
        )
        
        return hybrid_value
    
    def get_hybrid_contribution(self, client_id: int) -> float:
        """获取混合模式的贡献度值"""
        if client_id not in self.client_ema:
            return 0.0
        
        ema_value = self.get_ema(client_id)
        
        if client_id in self.client_recent_window and len(self.client_recent_window[client_id]) > 0:
            window_avg = np.mean(self.client_recent_window[client_id])
        else:
            window_avg = ema_value  # 窗口为空时使用EMA值
        
        hybrid_value = (
            self.hybrid_ema_weight * ema_value + 
            self.hybrid_window_weight * window_avg
        )
        
        return hybrid_value
    
    # =========================================================================
    # 统一接口方法 / Unified Interface Methods
    # =========================================================================
    
    def update_client_contribution(self, client_id: int, contribution: float, 
                                   round_num: int) -> float:
        """
        统一的贡献度更新接口 / Unified contribution update interface
        根据配置的模式自动选择合适的方法
        Automatically selects appropriate method based on configured mode
        
        Args:
            client_id: 客户端ID / Client ID
            contribution: 当前贡献度 / Current contribution
            round_num: 当前轮次 / Current round
            
        Returns:
            更新后的贡献度值 / Updated contribution value
        """
        # 记录历史（用于分析）
        self.contribution_history.append({
            'round': round_num,
            'client_id': client_id,
            'contribution': contribution,
            'method': self.smoothing_method
        })
        
        if self.smoothing_method == "ema":
            return self.update_ema(client_id, contribution, round_num)
        
        elif self.smoothing_method == "sliding_window":
            # 转换为积分（贡献度 × 1000）
            points = contribution * 1000
            self.update_client_slice_points(client_id, round_num, points)
            return self.get_active_points(client_id, round_num) / 1000  # 转回贡献度
        
        elif self.smoothing_method == "hybrid":
            return self.update_hybrid(client_id, contribution, round_num)
        
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")
    
    def get_client_weighted_contribution(self, client_id: int, current_round: int = 0) -> float:
        """
        获取客户端的加权贡献度 / Get client's weighted contribution
        统一接口，自动根据模式返回相应的值
        Unified interface, automatically returns appropriate value based on mode
        
        Args:
            client_id: 客户端ID / Client ID
            current_round: 当前轮次（滑动窗口模式需要）/ Current round (needed for sliding window)
            
        Returns:
            加权贡献度 / Weighted contribution
        """
        if self.smoothing_method == "ema":
            return self.get_ema(client_id)
        
        elif self.smoothing_method == "sliding_window":
            return self.get_active_points(client_id, current_round) / 1000
        
        elif self.smoothing_method == "hybrid":
            return self.get_hybrid_contribution(client_id)
        
        else:
            return 0.0
    
    # =========================================================================
    # 统计和分析方法 / Statistics and Analysis Methods
    # =========================================================================
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息 / Get statistics
        
        Returns:
            统计信息字典 / Statistics dictionary
        """
        stats = {
            'method': self.smoothing_method,
            'total_clients': len(self.client_ema) if self.smoothing_method == "ema" else len(self.client_slice_points)
        }
        
        if self.smoothing_method == "ema":
            ema_values = list(self.client_ema.values())
            stats['ema_stats'] = {
                'mean': np.mean(ema_values) if ema_values else 0,
                'std': np.std(ema_values) if ema_values else 0,
                'min': np.min(ema_values) if ema_values else 0,
                'max': np.max(ema_values) if ema_values else 0
            }
            stats['avg_participation'] = np.mean(list(self.client_participation_count.values())) if self.client_participation_count else 0
        
        elif self.smoothing_method == "sliding_window":
            all_points = []
            for client_points in self.client_slice_points.values():
                all_points.extend(client_points.values())
            
            if all_points:
                stats['window_stats'] = {
                    'mean': np.mean(all_points),
                    'std': np.std(all_points),
                    'total_slices': len(set(s for client_points in self.client_slice_points.values() for s in client_points.keys()))
                }
        
        elif self.smoothing_method == "hybrid":
            hybrid_values = [self.get_hybrid_contribution(cid) for cid in self.client_ema.keys()]
            if hybrid_values:
                stats['hybrid_stats'] = {
                    'mean': np.mean(hybrid_values),
                    'std': np.std(hybrid_values),
                    'ema_weight': self.hybrid_ema_weight,
                    'window_weight': self.hybrid_window_weight
                }
        
        return stats
    
    def get_client_evolution(self, client_id: int) -> List[Dict]:
        """
        获取客户端贡献度演化历史 / Get client contribution evolution history
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            历史记录列表 / History record list
        """
        client_history = [
            record for record in self.contribution_history 
            if record['client_id'] == client_id
        ]
        return client_history
    
    def set_system_activity(self, activity: float) -> None:
        """设置系统活跃度（用于动态时间片）"""
        self.system_activity = activity
    
    def get_memory_usage_estimate(self) -> Dict:
        """
        估算内存占用 / Estimate memory usage
        
        Returns:
            内存占用估算 / Memory usage estimate
        """
        if self.smoothing_method == "ema":
            # EMA: 每个客户端存储1个float (ema值) + 1个int (参与次数)
            client_data_size = len(self.client_ema) * (8 + 4)  # bytes
            memory_type = "O(N) - 每客户端1个值"
        
        elif self.smoothing_method == "sliding_window":
            # 滑动窗口: 每个客户端存储多个时间片数据
            total_slice_points = sum(len(slices) for slices in self.client_slice_points.values())
            client_data_size = total_slice_points * 8  # bytes
            memory_type = f"O(N×W) - 总计{total_slice_points}个值"
        
        else:  # hybrid
            # 混合模式: EMA数据 + 小窗口数据
            ema_size = len(self.client_ema) * 8
            window_size = sum(len(w) for w in self.client_recent_window.values()) * 8
            client_data_size = ema_size + window_size
            memory_type = "O(N×W_small) - EMA + 小窗口"
        
        return {
            'method': self.smoothing_method,
            'estimated_bytes': client_data_size,
            'estimated_kb': client_data_size / 1024,
            'memory_complexity': memory_type,
            'num_clients': len(self.client_ema) if self.smoothing_method != "sliding_window" else len(self.client_slice_points)
        }