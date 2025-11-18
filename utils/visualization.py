"""
utils/visualization.py (Complete with IPR Visualization)
可视化模块 - 完整版，包含IPR可视化
Visualization Module - Complete version with IPR visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'  # 或 'Liberation Sans'

# 如果需要显示中文，使用：
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Visualizer:
    """
    可视化工具 / Visualization Tool
    生成各类训练和评估图表，包括IPR可视化
    Generate various training and evaluation charts, including IPR visualization
    """
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """
        初始化 / Initialize
        
        Args:
            output_dir: 输出目录 / Output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图风格 / Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 字体设置 / Font settings
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"Visualizer initialized. Figures will be saved to: {output_dir}")
    
    def plot_training_curves(self, metrics_history: Dict, 
                            experiment_name: str,
                            save: bool = True) -> None:
        """
        绘制训练曲线 / Plot training curves
        包括准确率、贡献度、时间等
        Including accuracy, contribution, time, etc.
        
        Args:
            metrics_history: 指标历史数据 / Metrics history data
            experiment_name: 实验名称 / Experiment name
            save: 是否保存 / Whether to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Curves - {experiment_name}', fontsize=16, fontweight='bold')
        
        rounds = metrics_history['rounds']
        
        # 1. 准确率曲线 / Accuracy curves
        ax1 = axes[0, 0]
        ax1.plot(rounds, metrics_history['avg_client_accuracy'], 
                label='Avg Client Accuracy', linewidth=2, color='blue')
        ax1.plot(rounds, metrics_history['max_client_accuracy'], 
                label='Max Client Accuracy', linewidth=1.5, color='green', alpha=0.7)
        ax1.fill_between(rounds, 
                         metrics_history['avg_client_accuracy'],
                         metrics_history['max_client_accuracy'],
                         alpha=0.2, color='green')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Client Accuracy over Rounds', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 贡献度分布 / Contribution distribution
        ax2 = axes[0, 1]
        if 'contributions' in metrics_history and metrics_history['contributions']:
            # 计算每轮的平均贡献度和标准差
            avg_contributions = []
            std_contributions = []
            for round_contrib in metrics_history['contributions']:
                if round_contrib:
                    values = list(round_contrib.values())
                    avg_contributions.append(np.mean(values))
                    std_contributions.append(np.std(values))
                else:
                    avg_contributions.append(0)
                    std_contributions.append(0)
            
            ax2.plot(rounds, avg_contributions, label='Avg Contribution', 
                    linewidth=2, color='purple')
            ax2.fill_between(rounds,
                            np.array(avg_contributions) - np.array(std_contributions),
                            np.array(avg_contributions) + np.array(std_contributions),
                            alpha=0.3, color='purple', label='±1 Std')
            ax2.set_xlabel('Round', fontsize=12)
            ax2.set_ylabel('Normalized Contribution', fontsize=12)
            ax2.set_title('CGSV Contribution (Normalized)', fontsize=13, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No contribution data', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Contribution Distribution', fontsize=13, fontweight='bold')
        
        # 3. 原始CGSV分布 / Raw CGSV distribution
        ax3 = axes[1, 0]
        if 'raw_contributions' in metrics_history and metrics_history['raw_contributions']:
            avg_raw = []
            std_raw = []
            for round_contrib in metrics_history['raw_contributions']:
                if round_contrib:
                    values = list(round_contrib.values())
                    avg_raw.append(np.mean(values))
                    std_raw.append(np.std(values))
                else:
                    avg_raw.append(0)
                    std_raw.append(0)
            
            ax3.plot(rounds, avg_raw, label='Avg Raw CGSV', 
                    linewidth=2, color='orange')
            ax3.fill_between(rounds,
                            np.array(avg_raw) - np.array(std_raw),
                            np.array(avg_raw) + np.array(std_raw),
                            alpha=0.3, color='orange', label='±1 Std')
            ax3.set_xlabel('Round', fontsize=12)
            ax3.set_ylabel('Raw CGSV (Cosine Similarity)', fontsize=12)
            ax3.set_title('Raw CGSV over Rounds', fontsize=13, fontweight='bold')
            ax3.legend(loc='best', fontsize=10)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No raw contribution data', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Raw CGSV', fontsize=13, fontweight='bold')
        
        # 4. 每轮时间消耗 / Time per round
        ax4 = axes[1, 1]
        if 'time_per_round' in metrics_history and metrics_history['time_per_round']:
            ax4.bar(rounds, metrics_history['time_per_round'], 
                   alpha=0.7, color='teal', edgecolor='black', linewidth=0.5)
            ax4.axhline(y=np.mean(metrics_history['time_per_round']), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(metrics_history["time_per_round"]):.2f}s')
            ax4.set_xlabel('Round', fontsize=12)
            ax4.set_ylabel('Time (seconds)', fontsize=12)
            ax4.set_title('Time Consumption per Round', fontsize=13, fontweight='bold')
            ax4.legend(loc='upper right', fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'No time data', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Time Consumption', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 
                                   f'{experiment_name}_training_curves.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Training curves saved: {filepath}")
        
        plt.close()
    
    def plot_pcc_scatter(self, standalone_accuracies: List[float],
                        federated_accuracies: List[float],
                        pcc_value: float,
                        p_value: float,
                        experiment_name: str,
                        save: bool = True) -> None:
        """
        绘制PCC散点图 / Plot PCC scatter plot
        
        Args:
            standalone_accuracies: 独立训练准确率 / Standalone accuracies
            federated_accuracies: 联邦学习准确率 / Federated accuracies
            pcc_value: PCC值 / PCC value
            p_value: p值 / p-value
            experiment_name: 实验名称 / Experiment name
            save: 是否保存 / Whether to save
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 散点图 / Scatter plot
        scatter = ax.scatter(standalone_accuracies, federated_accuracies,
                           alpha=0.6, s=100, c=federated_accuracies,
                           cmap='viridis', edgecolors='black', linewidth=1)
        
        # 添加颜色条 / Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Federated Accuracy', fontsize=11)
        
        # 对角线 y=x / Diagonal line y=x
        min_val = min(min(standalone_accuracies), min(federated_accuracies))
        max_val = max(max(standalone_accuracies), max(federated_accuracies))
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, alpha=0.7, label='y=x (No improvement)')
        
        # 拟合线 / Fit line
        z = np.polyfit(standalone_accuracies, federated_accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.7, 
               label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # 标注PCC / Annotate PCC
        ax.text(0.05, 0.95, 
               f'PCC = {pcc_value:.4f}\np-value = {p_value:.4f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Standalone Accuracy', fontsize=13, fontweight='bold')
        ax.set_ylabel('Federated Learning Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'PCC Scatter Plot - {experiment_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 设置相同的坐标轴范围 / Set equal axis range
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 
                                   f'{experiment_name}_pcc_scatter.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ PCC scatter plot saved: {filepath}")
        
        plt.close()
    
    def plot_ipr_bar(self, ipr_data: Dict,
                     experiment_name: str,
                     save: bool = True) -> None:
        """
        绘制IPR柱状图（按客户端排序）/ Plot IPR bar chart (sorted by client)
        
        Args:
            ipr_data: IPR数据 / IPR data
            experiment_name: 实验名称 / Experiment name
            save: 是否保存 / Whether to save
        """
        if not ipr_data or 'improvements' not in ipr_data:
            print("No IPR data available for visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # 子图1: 改进程度柱状图 / Subplot 1: Improvement bar chart
        client_ids = ipr_data['client_ids']
        improvements = ipr_data['improvements']
        benefited = ipr_data['benefited']
        
        colors = ['green' if b else 'red' for b in benefited]
        
        bars = ax1.bar(range(len(client_ids)), improvements, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 零线 / Zero line
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        
        # 标注IPR / Annotate IPR
        ipr_text = (f"IPR = {ipr_data['ipr_value']:.4f} ({ipr_data['ipr_percentage']:.2f}%)\n"
                   f"Benefited: {ipr_data['num_benefited']}/{ipr_data['total_clients']}")
        ax1.text(0.02, 0.98, ipr_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax1.set_xlabel('Client ID (sorted by improvement)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy Improvement (Federated - Standalone)', fontsize=12, fontweight='bold')
        ax1.set_title('Client-wise Performance Improvement', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 设置x轴标签（每隔5个显示一次）
        if len(client_ids) > 20:
            tick_indices = list(range(0, len(client_ids), max(1, len(client_ids)//10)))
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([str(client_ids[i]) for i in tick_indices], 
                               rotation=45, ha='right')
        else:
            ax1.set_xticks(range(len(client_ids)))
            ax1.set_xticklabels([str(cid) for cid in client_ids], 
                               rotation=45, ha='right')
        
        # 子图2: 准确率对比 / Subplot 2: Accuracy comparison
        standalone_accs = ipr_data['standalone_accuracies']
        federated_accs = ipr_data['federated_accuracies']
        
        x = np.arange(len(client_ids))
        width = 0.35
        
        ax2.bar(x - width/2, standalone_accs, width, label='Standalone',
               alpha=0.8, color='skyblue', edgecolor='black', linewidth=0.5)
        ax2.bar(x + width/2, federated_accs, width, label='Federated',
               alpha=0.8, color='lightcoral', edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Client ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Standalone vs Federated Accuracy', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 设置x轴标签
        if len(client_ids) > 20:
            tick_indices = list(range(0, len(client_ids), max(1, len(client_ids)//10)))
            ax2.set_xticks(tick_indices)
            ax2.set_xticklabels([str(client_ids[i]) for i in tick_indices], 
                               rotation=45, ha='right')
        else:
            ax2.set_xticks(x)
            ax2.set_xticklabels([str(cid) for cid in client_ids], 
                               rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 
                                   f'{experiment_name}_ipr_bar.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ IPR bar chart saved: {filepath}")
        
        plt.close()
    
    def plot_ipr_history(self, ipr_history: List[float],
                        experiment_name: str,
                        save: bool = True) -> None:
        """
        绘制IPR历史曲线 / Plot IPR history curve
        
        Args:
            ipr_history: IPR历史数据 / IPR history data
            experiment_name: 实验名称 / Experiment name
            save: 是否保存 / Whether to save
        """
        if not ipr_history:
            print("No IPR history data available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rounds = list(range(1, len(ipr_history) + 1))
        
        # IPR曲线 / IPR curve
        ax.plot(rounds, ipr_history, linewidth=2.5, color='darkgreen', 
               marker='o', markersize=4, label='IPR per Round')
        
        # 平均线 / Average line
        avg_ipr = np.mean(ipr_history)
        ax.axhline(y=avg_ipr, color='red', linestyle='--', linewidth=2,
                  label=f'Average IPR: {avg_ipr:.4f}')
        
        # 最后10轮平均 / Last 10 rounds average
        if len(ipr_history) >= 10:
            final_10_avg = np.mean(ipr_history[-10:])
            ax.axhline(y=final_10_avg, color='blue', linestyle=':', linewidth=2,
                      label=f'Last 10 Rounds Avg: {final_10_avg:.4f}')
        
        # 目标线 / Target lines
        ax.axhline(y=0.95, color='gray', linestyle='-.', linewidth=1.5, 
                  alpha=0.5, label='Excellent (0.95)')
        ax.axhline(y=0.80, color='gray', linestyle='-.', linewidth=1.5, 
                  alpha=0.5, label='Good (0.80)')
        
        ax.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax.set_ylabel('IPR (Incentivized Participation Rate)', fontsize=13, fontweight='bold')
        ax.set_title(f'IPR Evolution over Training - {experiment_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 填充区域 / Fill areas
        ax.fill_between(rounds, 0.95, 1.0, alpha=0.1, color='green', label='_nolegend_')
        ax.fill_between(rounds, 0.80, 0.95, alpha=0.1, color='yellow', label='_nolegend_')
        ax.fill_between(rounds, 0, 0.80, alpha=0.1, color='red', label='_nolegend_')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 
                                   f'{experiment_name}_ipr_history.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ IPR history saved: {filepath}")
        
        plt.close()
    
    def plot_membership_distribution(self, membership_stats: Dict,
                                    experiment_name: str,
                                    save: bool = True) -> None:
        """
        绘制会员等级分布 / Plot membership level distribution
        
        Args:
            membership_stats: 会员统计数据 / Membership statistics
            experiment_name: 实验名称 / Experiment name
            save: 是否保存 / Whether to save
        """
        if not membership_stats or 'level_distribution' not in membership_stats:
            print("No membership data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        levels = ['Bronze', 'Silver', 'Gold', 'Diamond']
        counts = [membership_stats['level_distribution'].get(level.lower(), 0) 
                 for level in levels]
        percentages = [membership_stats['level_percentages'].get(level.lower(), 0) 
                      for level in levels]
        
        colors = ['#CD7F32', '#C0C0C0', '#FFD700', '#B9F2FF']
        
        # 子图1: 柱状图 / Subplot 1: Bar chart
        bars = ax1.bar(levels, counts, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # 在柱子上标注数量和百分比 / Annotate counts and percentages
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Membership Level', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Number of Clients', fontsize=13, fontweight='bold')
        ax1.set_title('Membership Level Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 子图2: 饼图 / Subplot 2: Pie chart
        wedges, texts, autotexts = ax2.pie(counts, labels=levels, colors=colors,
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # 美化饼图 / Beautify pie chart
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(12)
        
        ax2.set_title('Membership Level Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 
                                   f'{experiment_name}_membership_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Membership distribution saved: {filepath}")
        
        plt.close()
    
    def plot_comprehensive_summary(self, metrics: Dict,
                                  experiment_name: str,
                                  save: bool = True) -> None:
        """
        绘制综合摘要图 / Plot comprehensive summary
        包含所有关键指标的总览
        Overview of all key metrics
        
        Args:
            metrics: 指标数据 / Metrics data
            experiment_name: 实验名称 / Experiment name
            save: 是否保存 / Whether to save
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 标题 / Title
        fig.suptitle(f'Comprehensive Summary - {experiment_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 准确率统计 / Accuracy statistics
        ax1 = fig.add_subplot(gs[0, 0])
        acc_stats = metrics.get('client_accuracy', {})
        categories = ['Avg\nFinal', 'Max\nFinal', 'Min\nFinal', 'Avg\nMean']
        values = [acc_stats.get('avg_final', 0), acc_stats.get('max_final', 0),
                 acc_stats.get('min_final', 0), acc_stats.get('avg_mean', 0)]
        ax1.bar(categories, values, color=['blue', 'green', 'red', 'purple'], alpha=0.7)
        ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax1.set_title('Accuracy Statistics', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(values):
            ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. IPR指标 / IPR metrics
        ax2 = fig.add_subplot(gs[0, 1])
        ipr_stats = metrics.get('ipr', {})
        ipr_value = ipr_stats.get('final_ipr', 0)
        colors_ipr = ['green' if ipr_value >= 0.95 else 'yellow' if ipr_value >= 0.80 else 'red']
        ax2.bar(['IPR'], [ipr_value], color=colors_ipr, alpha=0.7, width=0.5)
        ax2.set_ylim([0, 1.0])
        ax2.set_ylabel('IPR Value', fontsize=11, fontweight='bold')
        ax2.set_title(f'IPR: {ipr_value:.4f} ({ipr_value*100:.2f}%)', 
                     fontsize=12, fontweight='bold')
        ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Excellent')
        ax2.axhline(y=0.80, color='yellow', linestyle='--', alpha=0.5, label='Good')
        ax2.legend(fontsize=8)
        ax2.text(0, ipr_value, f'{ipr_value:.4f}', ha='center', va='bottom', fontsize=11)
        
        # 3. PCC / PCC
        ax3 = fig.add_subplot(gs[0, 2])
        pcc_value = metrics.get('pcc', 0)
        colors_pcc = ['green' if abs(pcc_value) >= 0.8 else 'yellow' if abs(pcc_value) >= 0.5 else 'red']
        ax3.bar(['PCC'], [pcc_value], color=colors_pcc, alpha=0.7, width=0.5)
        ax3.set_ylim([-1.0, 1.0])
        ax3.set_ylabel('PCC Value', fontsize=11, fontweight='bold')
        ax3.set_title(f'PCC: {pcc_value:.4f}', fontsize=12, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.text(0, pcc_value, f'{pcc_value:.4f}', ha='center', 
                va='bottom' if pcc_value > 0 else 'top', fontsize=11)
        
        # 4. 性能提升分布 / Performance improvement distribution
        ax4 = fig.add_subplot(gs[1, :])
        if 'ipr_details' in metrics and 'client_improvements' in metrics['ipr_details']:
            improvements_dict = metrics['ipr_details']['client_improvements']
            improvements = [info['improvement'] for info in improvements_dict.values()]
            
            ax4.hist(improvements, bins=30, color='steelblue', alpha=0.7, 
                    edgecolor='black', linewidth=0.5)
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero improvement')
            ax4.axvline(x=np.mean(improvements), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(improvements):.4f}')
            ax4.set_xlabel('Accuracy Improvement', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Number of Clients', fontsize=11, fontweight='bold')
            ax4.set_title('Distribution of Performance Improvements', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 会员等级分布 / Membership distribution
        ax5 = fig.add_subplot(gs[2, 0])
        if 'membership_statistics' in metrics:
            mem_stats = metrics['membership_statistics']
            levels = ['Bronze', 'Silver', 'Gold', 'Diamond']
            percentages = [mem_stats['level_percentages'].get(level.lower(), 0) 
                          for level in levels]
            colors_mem = ['#CD7F32', '#C0C0C0', '#FFD700', '#B9F2FF']
            
            wedges, texts, autotexts = ax5.pie(percentages, labels=levels, colors=colors_mem,
                                               autopct='%1.1f%%', startangle=90)
            ax5.set_title('Membership\nDistribution', fontsize=11, fontweight='bold')
        
        # 6. 时间统计 / Time statistics
        ax6 = fig.add_subplot(gs[2, 1])
        time_stats = metrics.get('time_consumption', {})
        categories_time = ['Total', 'Mean', 'Max', 'Min']
        values_time = [time_stats.get('total', 0), time_stats.get('mean', 0),
                      time_stats.get('max', 0), time_stats.get('min', 0)]
        ax6.bar(categories_time, values_time, color=['purple', 'blue', 'green', 'red'], alpha=0.7)
        ax6.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax6.set_title('Time Consumption', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(values_time):
            ax6.text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # 7. 关键数值摘要 / Key metrics summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        summary_text = f"""
Key Metrics Summary

Clients: {metrics.get('num_clients', 0)}
Rounds: {metrics.get('num_rounds', 0)}

Accuracy:
  Final Avg: {acc_stats.get('avg_final', 0):.4f}
  Improvement: {acc_stats.get('avg_improvement', 0):.4f}

IPR:
  Final: {ipr_stats.get('final_ipr', 0):.4f}
  Benefited: {ipr_stats.get('num_benefited', 0)}/{metrics.get('num_clients', 0)}

PCC: {pcc_value:.4f}

Time:
  Total: {time_stats.get('total', 0):.1f}s
  Avg/Round: {time_stats.get('mean', 0):.2f}s
"""
        
        ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save:
            filepath = os.path.join(self.output_dir, 
                                   f'{experiment_name}_comprehensive_summary.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Comprehensive summary saved: {filepath}")
        
        plt.close()
    
    def generate_all_plots(self, metrics: Dict, 
                          metrics_history: Dict,
                          experiment_name: str) -> None:
        """
        生成所有图表 / Generate all plots
        
        Args:
            metrics: 最终指标 / Final metrics
            metrics_history: 指标历史 / Metrics history
            experiment_name: 实验名称 / Experiment name
        """
        print(f"\nGenerating all visualizations for {experiment_name}...")
        
        # 1. 训练曲线 / Training curves
        self.plot_training_curves(metrics_history, experiment_name)
        
        # 2. PCC散点图 / PCC scatter plot
        pcc_data = metrics.get('pcc_details', {})
        if 'data_points' in pcc_data:
            standalone = [p[0] for p in pcc_data['data_points']]
            federated = [p[1] for p in pcc_data['data_points']]
            self.plot_pcc_scatter(standalone, federated, 
                                metrics['pcc'], pcc_data.get('p_value', 0),
                                experiment_name)
        
        # 3. IPR柱状图 / IPR bar chart
        ipr_vis_data = metrics.get('ipr_details', {})
        if 'client_improvements' in ipr_vis_data:
            # 准备可视化数据
            sorted_clients = sorted(ipr_vis_data['client_improvements'].items(),
                                  key=lambda x: x[1]['improvement'], reverse=True)
            
            ipr_plot_data = {
                'client_ids': [cid for cid, _ in sorted_clients],
                'improvements': [info['improvement'] for _, info in sorted_clients],
                'benefited': [info['benefited'] for _, info in sorted_clients],
                'standalone_accuracies': [info['standalone'] for _, info in sorted_clients],
                'federated_accuracies': [info['federated'] for _, info in sorted_clients],
                'ipr_value': ipr_vis_data['ipr_accuracy'],
                'ipr_percentage': ipr_vis_data['benefited_percentage'],
                'num_benefited': ipr_vis_data['num_benefited'],
                'total_clients': ipr_vis_data['total_clients']
            }
            self.plot_ipr_bar(ipr_plot_data, experiment_name)
        
        # 4. IPR历史曲线 / IPR history
        if 'ipr' in metrics and 'ipr_history' in metrics['ipr']:
            self.plot_ipr_history(metrics['ipr']['ipr_history'], experiment_name)
        
        # 5. 会员等级分布 / Membership distribution
        if 'membership_statistics' in metrics:
            self.plot_membership_distribution(metrics['membership_statistics'], 
                                            experiment_name)
        
        # 6. 综合摘要 / Comprehensive summary
        self.plot_comprehensive_summary(metrics, experiment_name)
        
        print(f"✓ All visualizations completed for {experiment_name}")