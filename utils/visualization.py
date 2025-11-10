"""
utils/visualization.py
可视化模块 - 支持激励机制评价指标可视化
Visualization Module - Supporting Incentive Mechanism Metrics Visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import os


class Visualizer:
    """
    可视化工具 / Visualization Tool
    生成联邦学习实验的各种图表，包括激励机制评价指标
    Generate various charts for federated learning experiments, including incentive mechanism metrics
    """
    
    def __init__(self, save_dir: str = "outputs/results/plots"):
        """
        初始化可视化器 / Initialize visualizer
        
        Args:
            save_dir: 图表保存目录 / Chart save directory
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格 / Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_incentive_metrics(self, metrics_history: Dict, 
                              experiment_name: str) -> None:
        """
        绘制激励机制评价指标 / Plot incentive mechanism evaluation metrics
        
        Args:
            metrics_history: 指标历史 / Metrics history
            experiment_name: 实验名称 / Experiment name
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rounds = metrics_history.get('round_times', 
                                    list(range(len(metrics_history.get('contribution_reward_correlation', [])))))
        
        # 1. 贡献-回报相关性 / Contribution-Reward Correlation
        if 'contribution_reward_correlation' in metrics_history:
            corr_values = metrics_history['contribution_reward_correlation']
            axes[0, 0].plot(rounds[:len(corr_values)], corr_values, 'b-', linewidth=2, marker='o', markersize=4)
            axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].fill_between(rounds[:len(corr_values)], 0, corr_values, 
                                   where=[v > 0 for v in corr_values], 
                                   color='green', alpha=0.2, label='Positive Correlation')
            axes[0, 0].fill_between(rounds[:len(corr_values)], 0, corr_values, 
                                   where=[v < 0 for v in corr_values], 
                                   color='red', alpha=0.2, label='Negative Correlation')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Pearson Correlation Coefficient')
            axes[0, 0].set_title('Contribution-Reward Correlation\n(贡献-回报相关性)')
            axes[0, 0].set_ylim([-1, 1])
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 公平性指数 / Fairness Index
        if 'fairness_index' in metrics_history:
            fairness_values = metrics_history['fairness_index']
            axes[0, 1].plot(rounds[:len(fairness_values)], fairness_values, 'g-', linewidth=2, marker='s', markersize=4)
            axes[0, 1].axhline(y=1.0, color='gold', linestyle='--', alpha=0.7, label='Perfect Fairness')
            axes[0, 1].fill_between(rounds[:len(fairness_values)], 0, fairness_values, 
                                   color='green', alpha=0.3)
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel("Jain's Fairness Index")
            axes[0, 1].set_title('Reward Distribution Fairness\n(奖励分配公平性)')
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 激励有效性 / Incentive Effectiveness
        if 'incentive_effectiveness' in metrics_history:
            effectiveness_values = metrics_history['incentive_effectiveness']
            axes[1, 0].plot(rounds[:len(effectiveness_values)], effectiveness_values, 
                          'm-', linewidth=2, marker='^', markersize=4)
            axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Neutral')
            axes[1, 0].fill_between(rounds[:len(effectiveness_values)], 0.5, effectiveness_values,
                                   where=[v > 0.5 for v in effectiveness_values],
                                   color='purple', alpha=0.2, label='Positive Effect')
            axes[1, 0].fill_between(rounds[:len(effectiveness_values)], 0.5, effectiveness_values,
                                   where=[v < 0.5 for v in effectiveness_values],
                                   color='orange', alpha=0.2, label='Negative Effect')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Effectiveness Score')
            axes[1, 0].set_title('Incentive Mechanism Effectiveness\n(激励机制有效性)')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 综合指标雷达图 / Comprehensive Metrics Radar Chart
        if all(k in metrics_history for k in ['contribution_reward_correlation', 
                                               'fairness_index', 'incentive_effectiveness']):
            # 获取最终值 / Get final values
            final_corr = metrics_history['contribution_reward_correlation'][-1] if metrics_history['contribution_reward_correlation'] else 0
            final_fair = metrics_history['fairness_index'][-1] if metrics_history['fairness_index'] else 0
            final_eff = metrics_history['incentive_effectiveness'][-1] if metrics_history['incentive_effectiveness'] else 0
            
            # 准备雷达图数据 / Prepare radar chart data
            categories = ['Contribution-Reward\nCorrelation', 'Fairness\nIndex', 
                         'Incentive\nEffectiveness']
            values = [(final_corr + 1) / 2, final_fair, final_eff]  # 归一化到[0,1]
            
            # 创建雷达图 / Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形 / Close the shape
            angles += angles[:1]
            
            ax_radar = plt.subplot(224, projection='polar')
            ax_radar.plot(angles, values, 'o-', linewidth=2, color='red', markersize=8)
            ax_radar.fill(angles, values, color='red', alpha=0.25)
            ax_radar.set_theta_offset(np.pi / 2)
            ax_radar.set_theta_direction(-1)
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax_radar.set_title('Final Incentive Metrics Summary\n(最终激励指标总结)', pad=20)
            ax_radar.grid(True)
        
        plt.suptitle(f'Incentive Mechanism Evaluation Metrics - {experiment_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_incentive_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Incentive metrics plot saved to {save_path}")
    
    def plot_contribution_performance_scatter(self, contributions: List[Dict],
                                             performances: List[Dict],
                                             experiment_name: str) -> None:
        """
        绘制贡献度与性能的散点图 / Plot contribution vs performance scatter plot
        
        Args:
            contributions: 贡献度历史 / Contribution history
            performances: 性能历史 / Performance history
            experiment_name: 实验名称 / Experiment name
        """
        if not contributions or not performances:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 选择初期、中期、末期的数据 / Select data from early, middle, and late stages
        stages = {
            'Early Stage': min(5, len(contributions) - 1),
            'Middle Stage': len(contributions) // 2,
            'Late Stage': len(contributions) - 1
        }
        
        for idx, (stage_name, round_idx) in enumerate(stages.items()):
            if round_idx < len(contributions) and round_idx < len(performances):
                contrib_dict = contributions[round_idx]
                perf_dict = performances[round_idx]
                
                # 获取共同的客户端 / Get common clients
                common_clients = set(contrib_dict.keys()) & set(perf_dict.keys())
                
                x_values = [contrib_dict[cid] for cid in common_clients]
                y_values = [perf_dict[cid] for cid in common_clients]
                
                # 绘制散点图 / Plot scatter
                axes[idx].scatter(x_values, y_values, alpha=0.6, s=50, c=x_values, cmap='viridis')
                
                # 添加趋势线 / Add trend line
                if len(x_values) > 1:
                    z = np.polyfit(x_values, y_values, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x_values), max(x_values), 100)
                    axes[idx].plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2)
                
                axes[idx].set_xlabel('Contribution (贡献度)')
                axes[idx].set_ylabel('Model Performance (模型性能)')
                axes[idx].set_title(f'{stage_name} (Round {round_idx + 1})')
                axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(f'Contribution-Performance Relationship Evolution - {experiment_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_contribution_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Contribution-performance scatter plot saved to {save_path}")
    
    def plot_training_curves(self, metrics_history: Dict, 
                            experiment_name: str) -> None:
        """
        绘制训练曲线 / Plot training curves
        
        Args:
            metrics_history: 指标历史 / Metrics history
            experiment_name: 实验名称 / Experiment name
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        rounds = metrics_history.get('round_times', list(range(len(metrics_history['accuracy']))))
        
        # 1. 准确率曲线 / Accuracy curve
        axes[0, 0].plot(rounds, metrics_history['accuracy'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy over Rounds')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 损失曲线 / Loss curve
        axes[0, 1].plot(rounds, metrics_history['loss'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Model Loss over Rounds')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 参与率曲线 / Participation rate curve
        axes[0, 2].plot(rounds, metrics_history['participation_rate'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Round')
        axes[0, 2].set_ylabel('Participation Rate')
        axes[0, 2].set_title('Client Participation Rate')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 系统活跃度曲线 / System activity curve
        axes[1, 0].plot(rounds, metrics_history['system_activity'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('System Activity')
        axes[1, 0].set_title('System Activity Level')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 模型质量差距 / Model quality gap
        if 'quality_gap' in metrics_history and metrics_history['quality_gap']:
            gap_rounds = list(range(len(metrics_history['quality_gap'])))
            axes[1, 1].plot(gap_rounds, metrics_history['quality_gap'], 'c-', linewidth=2)
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Quality Gap')
            axes[1, 1].set_title('Model Quality Gap\n(High vs Low Contributors)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 激励指标综合 / Combined incentive metrics
        if 'contribution_reward_correlation' in metrics_history:
            ax2 = axes[1, 2]
            rounds_incentive = list(range(len(metrics_history['contribution_reward_correlation'])))
            
            # 主Y轴：相关性 / Primary Y-axis: Correlation
            color = 'tab:blue'
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Correlation', color=color)
            ax2.plot(rounds_incentive, metrics_history['contribution_reward_correlation'], 
                    color=color, linewidth=2, label='Correlation')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([-1, 1])
            
            # 次Y轴：公平性 / Secondary Y-axis: Fairness
            if 'fairness_index' in metrics_history:
                ax3 = ax2.twinx()
                color = 'tab:green'
                ax3.set_ylabel('Fairness Index', color=color)
                ax3.plot(rounds_incentive[:len(metrics_history['fairness_index'])], 
                        metrics_history['fairness_index'], 
                        color=color, linewidth=2, linestyle='--', label='Fairness')
                ax3.tick_params(axis='y', labelcolor=color)
                ax3.set_ylim([0, 1])
            
            ax2.set_title('Incentive Metrics Overview')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress - {experiment_name}', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def create_comprehensive_report(self, metrics_summary: Dict, 
                                   experiment_name: str) -> None:
        """
        创建综合报告 / Create comprehensive report
        
        Args:
            metrics_summary: 指标摘要 / Metrics summary
            experiment_name: 实验名称 / Experiment name
        """
        fig = plt.figure(figsize=(20, 14))
        
        # 创建网格布局 / Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 模型性能总结 / Model performance summary
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = ['Accuracy', 'Loss', 'Participation Rate', 'System Activity']
        final_values = [
            metrics_summary.get('accuracy_final', 0),
            metrics_summary.get('loss_final', 0),
            metrics_summary.get('participation_rate_final', 0),
            metrics_summary.get('system_activity_final', 0)
        ]
        colors = ['green', 'red', 'blue', 'purple']
        bars = ax1.bar(metrics, final_values, color=colors, alpha=0.7)
        ax1.set_ylabel('Value')
        ax1.set_title('Final Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签 / Add value labels
        for bar, val in zip(bars, final_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 2. 激励机制指标总结 / Incentive mechanism metrics summary
        ax2 = fig.add_subplot(gs[0, 2])
        incentive_metrics = {
            'Contribution-Reward\nCorrelation': metrics_summary.get('contribution_reward_correlation_final', 0),
            'Fairness\nIndex': metrics_summary.get('fairness_index_final', 0),
            'Incentive\nEffectiveness': metrics_summary.get('incentive_effectiveness_final', 0)
        }
        
        wedges, texts, autotexts = ax2.pie(
            [abs(v) for v in incentive_metrics.values()],
            labels=incentive_metrics.keys(),
            autopct='%1.1f%%',
            colors=['#FF9999', '#66B2FF', '#99FF99']
        )
        ax2.set_title('Incentive Metrics Distribution', fontsize=14, fontweight='bold')
        
        # 3. 收敛信息 / Convergence information
        ax3 = fig.add_subplot(gs[1, 0])
        convergence_round = metrics_summary.get('convergence_round', -1)
        ax3.text(0.5, 0.5, f'Convergence Round\n\n{convergence_round}',
                ha='center', va='center', fontsize=24, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Training Convergence', fontsize=14, fontweight='bold')
        
        # 4. 关键指标对比表 / Key metrics comparison table
        ax4 = fig.add_subplot(gs[1, 1:])
        table_data = []
        metrics_to_show = [
            ('Accuracy', 'accuracy'),
            ('Loss', 'loss'),
            ('Participation Rate', 'participation_rate'),
            ('System Activity', 'system_activity'),
            ('Contribution-Reward Corr.', 'contribution_reward_correlation'),
            ('Fairness Index', 'fairness_index'),
            ('Incentive Effectiveness', 'incentive_effectiveness')
        ]
        
        for metric_name, metric_key in metrics_to_show:
            row = [
                metric_name,
                f"{metrics_summary.get(f'{metric_key}_final', 0):.4f}",
                f"{metrics_summary.get(f'{metric_key}_avg', 0):.4f}",
                f"{metrics_summary.get(f'{metric_key}_max', 0):.4f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Metric', 'Final', 'Average', 'Maximum'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.35, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # 设置表格样式 / Set table style
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
        
        ax4.axis('off')
        ax4.set_title('Detailed Metrics Statistics', fontsize=14, fontweight='bold')
        
        # 5. 质量差距趋势 / Quality gap trend
        ax5 = fig.add_subplot(gs[2, :])
        ax5.text(0.5, 0.7, f"Average Quality Gap: {metrics_summary.get('quality_gap_avg', 0):.4f}",
                ha='center', fontsize=14, fontweight='bold')
        ax5.text(0.5, 0.5, f"Maximum Quality Gap: {metrics_summary.get('quality_gap_max', 0):.4f}",
                ha='center', fontsize=14)
        ax5.text(0.5, 0.3, f"Final Correlation: {metrics_summary.get('contribution_reward_correlation_final', 0):.4f}",
                ha='center', fontsize=14)
        ax5.text(0.5, 0.1, f"Average Fairness: {metrics_summary.get('fairness_index_avg', 0):.4f}",
                ha='center', fontsize=14)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Incentive Mechanism Performance Summary', fontsize=14, fontweight='bold')
        
        # 设置总标题 / Set main title
        plt.suptitle(f'Comprehensive Experiment Report - {experiment_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_comprehensive_report.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comprehensive report saved to {save_path}")