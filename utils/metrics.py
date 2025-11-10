"""
utils/metrics.py
评估指标模块 - 包含激励机制评价指标
Evaluation Metrics Module - Including Incentive Mechanism Evaluation Metrics
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats


class IncentiveMetricsCalculator:
    """
    激励机制评价指标计算器 / Incentive Mechanism Metrics Calculator
    计算激励机制的效果评价指标
    Calculate evaluation metrics for incentive mechanism effectiveness
    """
    
    def __init__(self):
        """初始化激励指标计算器 / Initialize incentive metrics calculator"""
        self.contribution_history = {}
        self.reward_history = {}
        self.performance_history = {}
    
    def calculate_contribution_reward_correlation(self, 
                                                 contributions: Dict[int, float],
                                                 performances: Dict[int, float]) -> float:
        """
        计算贡献-回报相关性 (Contribution-Reward Correlation)
        衡量客户端贡献度与其获得的模型性能之间的相关性
        
        Args:
            contributions: 客户端贡献度字典 / Client contribution dictionary
            performances: 客户端模型性能字典 / Client model performance dictionary
            
        Returns:
            皮尔逊相关系数 / Pearson correlation coefficient
        """
        # 确保有相同的客户端 / Ensure same clients
        common_clients = set(contributions.keys()) & set(performances.keys())
        
        if len(common_clients) < 2:
            return 0.0
        
        # 提取对应的值 / Extract corresponding values
        contrib_values = [contributions[cid] for cid in common_clients]
        perf_values = [performances[cid] for cid in common_clients]
        
        # 计算皮尔逊相关系数 / Calculate Pearson correlation
        if len(set(contrib_values)) == 1 or len(set(perf_values)) == 1:
            # 如果所有值相同，相关性为0 / If all values are same, correlation is 0
            return 0.0
        
        correlation, p_value = stats.pearsonr(contrib_values, perf_values)
        
        # 记录历史 / Record history
        self.contribution_history[datetime.now()] = contributions
        self.performance_history[datetime.now()] = performances
        
        return correlation
    
    def calculate_fairness_index(self, 
                                contributions: Dict[int, float],
                                rewards: Dict[int, float]) -> float:
        """
        计算公平性指数 (Fairness Index)
        使用Jain's Fairness Index评估奖励分配的公平性
        
        Args:
            contributions: 客户端贡献度字典 / Client contribution dictionary
            rewards: 客户端奖励字典 / Client rewards dictionary
            
        Returns:
            公平性指数 (0-1) / Fairness index (0-1)
        """
        if not contributions or not rewards:
            return 0.0
        
        # 计算贡献-奖励比率 / Calculate contribution-reward ratios
        ratios = []
        for client_id in contributions.keys():
            if client_id in rewards and contributions[client_id] > 0:
                ratio = rewards[client_id] / contributions[client_id]
                ratios.append(ratio)
        
        if not ratios:
            return 0.0
        
        # 计算Jain's Fairness Index
        # J = (sum(x_i))^2 / (n * sum(x_i^2))
        n = len(ratios)
        sum_ratios = sum(ratios)
        sum_squares = sum(r**2 for r in ratios)
        
        if sum_squares == 0:
            return 0.0
        
        fairness_index = (sum_ratios ** 2) / (n * sum_squares)
        
        return fairness_index
    
    def calculate_incentive_effectiveness(self, 
                                         participation_rates: List[float],
                                         system_activities: List[float],
                                         model_improvements: List[float]) -> float:
        """
        计算激励有效性 (Incentive Effectiveness)
        综合评估激励机制对系统整体性能的提升效果
        
        Args:
            participation_rates: 参与率历史 / Participation rate history
            system_activities: 系统活跃度历史 / System activity history
            model_improvements: 模型改进历史 / Model improvement history
            
        Returns:
            激励有效性得分 (0-1) / Incentive effectiveness score (0-1)
        """
        if not participation_rates or not system_activities or not model_improvements:
            return 0.0
        
        # 计算各指标的改进趋势 / Calculate improvement trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        
        # 计算趋势 / Calculate trends
        participation_trend = calculate_trend(participation_rates)
        activity_trend = calculate_trend(system_activities)
        improvement_trend = calculate_trend(model_improvements)
        
        # 归一化趋势值 / Normalize trend values
        max_trend = max(abs(participation_trend), abs(activity_trend), abs(improvement_trend), 0.001)
        
        norm_participation = participation_trend / max_trend
        norm_activity = activity_trend / max_trend
        norm_improvement = improvement_trend / max_trend
        
        # 计算综合效果分数 / Calculate composite effectiveness score
        # 权重：参与率30%，活跃度30%，模型改进40%
        effectiveness = 0.3 * norm_participation + 0.3 * norm_activity + 0.4 * norm_improvement
        
        # 映射到[0,1]范围 / Map to [0,1] range
        effectiveness_score = (effectiveness + 1) / 2
        
        return np.clip(effectiveness_score, 0, 1)


class MetricsCalculator:
    """
    评估指标计算器 / Metrics Calculator
    计算联邦学习和激励机制的各种指标
    Calculate various metrics for federated learning and incentive mechanisms
    """
    
    def __init__(self):
        """初始化指标计算器 / Initialize metrics calculator"""
        self.metrics_history = {
            'accuracy': [],
            'loss': [],
            'participation_rate': [],
            'system_activity': [],
            'client_participation': {},
            'level_distribution': [],
            'points_distribution': [],
            'round_times': [],
            'quality_gap': [],
            # 新增激励机制评价指标 / New incentive mechanism metrics
            'contribution_reward_correlation': [],
            'fairness_index': [],
            'incentive_effectiveness': [],
            'client_contributions': [],
            'client_performances': []
        }
        
        # 激励指标计算器 / Incentive metrics calculator
        self.incentive_calculator = IncentiveMetricsCalculator()
    
    def update_incentive_metrics(self, round_num: int,
                                contributions: Dict[int, float],
                                performances: Dict[int, float],
                                rewards: Dict[int, float]) -> None:
        """
        更新激励机制相关指标 / Update incentive mechanism related metrics
        
        Args:
            round_num: 轮次 / Round number
            contributions: 客户端贡献度 / Client contributions
            performances: 客户端模型性能 / Client model performances
            rewards: 客户端获得的奖励（积分）/ Client rewards (points)
        """
        # 计算贡献-回报相关性 / Calculate contribution-reward correlation
        correlation = self.incentive_calculator.calculate_contribution_reward_correlation(
            contributions, performances
        )
        self.metrics_history['contribution_reward_correlation'].append(correlation)
        
        # 计算公平性指数 / Calculate fairness index
        fairness = self.incentive_calculator.calculate_fairness_index(
            contributions, rewards
        )
        self.metrics_history['fairness_index'].append(fairness)
        
        # 计算激励有效性 / Calculate incentive effectiveness
        if len(self.metrics_history['accuracy']) > 0:
            # 使用准确率作为模型改进指标 / Use accuracy as model improvement metric
            model_improvements = self.metrics_history['accuracy']
        else:
            model_improvements = []
        
        effectiveness = self.incentive_calculator.calculate_incentive_effectiveness(
            self.metrics_history['participation_rate'],
            self.metrics_history['system_activity'],
            model_improvements
        )
        self.metrics_history['incentive_effectiveness'].append(effectiveness)
        
        # 记录贡献度和性能 / Record contributions and performances
        self.metrics_history['client_contributions'].append(contributions)
        self.metrics_history['client_performances'].append(performances)
    
    def calculate_participation_rate(self, participating_clients: int, 
                                    total_clients: int) -> float:
        """
        计算参与率 / Calculate participation rate
        
        Args:
            participating_clients: 参与的客户端数 / Number of participating clients
            total_clients: 总客户端数 / Total number of clients
            
        Returns:
            参与率 / Participation rate
        """
        if total_clients == 0:
            return 0
        return participating_clients / total_clients
    
    def calculate_client_participation(self, client_id: int, 
                                      rounds_participated: int,
                                      total_rounds: int,
                                      contribution: float) -> float:
        """
        计算单个客户端的参与度 / Calculate individual client participation
        
        Args:
            client_id: 客户端ID / Client ID
            rounds_participated: 参与的轮次数 / Number of rounds participated
            total_rounds: 总轮次数 / Total number of rounds
            contribution: 客户端贡献度 / Client contribution
            
        Returns:
            客户端参与度 / Client participation
        """
        # 参与频率 / Participation frequency
        frequency = rounds_participated / total_rounds if total_rounds > 0 else 0
        
        # 综合参与度 = 频率和贡献度的加权平均 / Combined participation
        participation = 0.4 * frequency + 0.6 * contribution
        
        return participation
    
    def calculate_system_activity(self, client_participations: Dict[int, float],
                                 active_clients: int,
                                 total_clients: int) -> float:
        """
        计算系统活跃度 / Calculate system activity
        
        Args:
            client_participations: 客户端参与度字典 / Client participation dictionary
            active_clients: 活跃客户端数 / Number of active clients
            total_clients: 总客户端数 / Total number of clients
            
        Returns:
            系统活跃度 / System activity
        """
        if total_clients == 0:
            return 0
        
        # 活跃客户端比例 / Active client ratio
        active_ratio = active_clients / total_clients
        
        # 平均参与度 / Average participation
        if len(client_participations) > 0:
            avg_participation = sum(client_participations.values()) / len(client_participations)
        else:
            avg_participation = 0
        
        # 综合系统活跃度 / Combined system activity
        system_activity = 0.6 * active_ratio + 0.4 * avg_participation
        
        return system_activity
    
    def update_metrics(self, round_num: int, accuracy: float, loss: float,
                      participation_rate: float, system_activity: float,
                      level_distribution: Dict, points_stats: Dict) -> None:
        """
        更新指标历史 / Update metrics history
        
        Args:
            round_num: 轮次 / Round number
            accuracy: 准确率 / Accuracy
            loss: 损失 / Loss
            participation_rate: 参与率 / Participation rate
            system_activity: 系统活跃度 / System activity
            level_distribution: 等级分布 / Level distribution
            points_stats: 积分统计 / Points statistics
        """
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['loss'].append(loss)
        self.metrics_history['participation_rate'].append(participation_rate)
        self.metrics_history['system_activity'].append(system_activity)
        self.metrics_history['level_distribution'].append(level_distribution)
        self.metrics_history['points_distribution'].append(points_stats)
        self.metrics_history['round_times'].append(round_num)
    
    def record_quality_gap(self, gap: float):
        """
        记录模型质量差距 / Record model quality gap
        
        Args:
            gap: 最高贡献度与最低贡献度客户端的模型准确率差距
        """
        self.metrics_history['quality_gap'].append(gap)
    
    def get_metrics_summary(self) -> Dict:
        """
        获取指标摘要 / Get metrics summary
        
        Returns:
            指标摘要字典 / Metrics summary dictionary
        """
        summary = {}
        
        # 基础指标统计 / Basic metrics statistics
        for key in ['accuracy', 'loss', 'participation_rate', 'system_activity']:
            if self.metrics_history[key]:
                summary[f'{key}_final'] = self.metrics_history[key][-1]
                summary[f'{key}_avg'] = np.mean(self.metrics_history[key])
                summary[f'{key}_max'] = np.max(self.metrics_history[key])
                summary[f'{key}_min'] = np.min(self.metrics_history[key])
        
        # 激励机制指标统计 / Incentive mechanism metrics statistics
        for key in ['contribution_reward_correlation', 'fairness_index', 'incentive_effectiveness']:
            if self.metrics_history[key]:
                summary[f'{key}_final'] = self.metrics_history[key][-1]
                summary[f'{key}_avg'] = np.mean(self.metrics_history[key])
                summary[f'{key}_max'] = np.max(self.metrics_history[key])
        
        # 质量差距统计 / Quality gap statistics
        if self.metrics_history['quality_gap']:
            summary['quality_gap_final'] = self.metrics_history['quality_gap'][-1]
            summary['quality_gap_avg'] = np.mean(self.metrics_history['quality_gap'])
            summary['quality_gap_max'] = np.max(self.metrics_history['quality_gap'])
        
        return summary
    
    def calculate_convergence_round(self, threshold: float = 0.95) -> int:
        """
        计算收敛轮次 / Calculate convergence round
        
        Args:
            threshold: 收敛阈值 / Convergence threshold
            
        Returns:
            收敛轮次 / Convergence round
        """
        if not self.metrics_history['accuracy']:
            return -1
        
        max_acc = max(self.metrics_history['accuracy'])
        target_acc = max_acc * threshold
        
        for i, acc in enumerate(self.metrics_history['accuracy']):
            if acc >= target_acc:
                return i
        
        return -1