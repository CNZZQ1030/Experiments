"""
utils/visualization.py
可视化模块 / Visualization Module
展示客户端在各自测试集上的性能
Show client performance on their own test sets
"""

from typing import Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """可视化工具 / Visualization Tool"""
    
    def __init__(self, save_dir: str = "outputs/plots"):
        """初始化 / Initialize"""
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置绘图风格 / Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_pcc_scatter(self, standalone_accuracies: List[float],
                        federated_accuracies: List[float],
                        pcc_value: float, p_value: float,
                        experiment_name: str) -> None:
        """
        绘制PCC散点图 / Plot PCC scatter plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：散点图+回归线 / Left: Scatter + regression
        ax1.scatter(standalone_accuracies, federated_accuracies, 
                   alpha=0.6, s=50, c='blue', edgecolors='navy', linewidth=1.5)
        
        # 回归线 / Regression line
        z = np.polyfit(standalone_accuracies, federated_accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(standalone_accuracies), max(standalone_accuracies), 100)
        ax1.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2, label='Regression')
        
        # 对角线 / Diagonal
        min_val = min(min(standalone_accuracies), min(federated_accuracies))
        max_val = max(max(standalone_accuracies), max(federated_accuracies))
        ax1.plot([min_val, max_val], [min_val, max_val], 'g--', 
                alpha=0.5, linewidth=1.5, label='Perfect correlation')
        
        ax1.set_xlabel('Standalone Accuracy (Test Set)', fontsize=12)
        ax1.set_ylabel('Federated Accuracy (Test Set)', fontsize=12)
        ax1.set_title(f'Correlation Analysis\nPCC = {pcc_value:.4f}, p = {p_value:.4f}', fontsize=13)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([min_val - 0.05, max_val + 0.05])
        ax1.set_ylim([min_val - 0.05, max_val + 0.05])
        
        # 右图：改进分布 / Right: Improvement distribution
        improvements = [f - s for s, f in zip(standalone_accuracies, federated_accuracies)]
        
        ax2.hist(improvements, bins=15, color='green', alpha=0.7, edgecolor='darkgreen')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax2.axvline(x=np.mean(improvements), color='blue', linestyle='-', 
                   linewidth=2, label=f'Mean: {np.mean(improvements):.4f}')
        
        ax2.set_xlabel('Accuracy Improvement', fontsize=12)
        ax2.set_ylabel('Number of Clients', fontsize=12)
        ax2.set_title('Performance Improvement Distribution', fontsize=13)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'PCC Analysis - {experiment_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_pcc_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCC plot saved: {save_path}")
    
    def plot_training_curves(self, metrics_history: Dict, experiment_name: str) -> None:
        """绘制训练曲线 / Plot training curves"""
        rounds = metrics_history.get('rounds', [])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 准确率曲线 / Accuracy curves
        if 'avg_client_accuracy' in metrics_history and metrics_history['avg_client_accuracy']:
            axes[0, 0].plot(rounds, metrics_history['avg_client_accuracy'], 
                          'b-', linewidth=2.5, label='Avg', marker='o', markersize=5)
        
        if 'max_client_accuracy' in metrics_history and metrics_history['max_client_accuracy']:
            axes[0, 0].plot(rounds, metrics_history['max_client_accuracy'], 
                          'r-', linewidth=2.5, label='Max', marker='^', markersize=5)
        
        axes[0, 0].set_xlabel('Round', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy (Test Sets)', fontsize=12)
        axes[0, 0].set_title('Client Accuracy Evolution', fontsize=13, fontweight='bold')
        axes[0, 0].legend(loc='best', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率差异 / Accuracy gap
        if 'avg_client_accuracy' in metrics_history and 'max_client_accuracy' in metrics_history:
            if metrics_history['avg_client_accuracy'] and metrics_history['max_client_accuracy']:
                gap = [m - a for m, a in zip(metrics_history['max_client_accuracy'], 
                                             metrics_history['avg_client_accuracy'])]
                
                axes[0, 1].plot(rounds, gap, 'purple', linewidth=2, marker='s', markersize=4)
                axes[0, 1].set_xlabel('Round', fontsize=12)
                axes[0, 1].set_ylabel('Accuracy Gap', fontsize=12)
                axes[0, 1].set_title('Performance Gap (Max - Avg)', fontsize=13, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                
                if len(gap) > 1:
                    z = np.polyfit(rounds, gap, 1)
                    p = np.poly1d(z)
                    axes[0, 1].plot(rounds, p(rounds), "r--", alpha=0.5, 
                                  label=f'Trend: {"↓" if z[0] < 0 else "↑"}')
                    axes[0, 1].legend(loc='best')
        
        # 贡献度分布 / Contribution distribution
        if 'contributions' in metrics_history and metrics_history['contributions']:
            contributions = metrics_history['contributions']
            contribution_data = []
            valid_rounds = []
            
            for i, round_contrib in enumerate(contributions):
                values = []
                if isinstance(round_contrib, dict):
                    values = [v for v in round_contrib.values() if v is not None and v > 0]
                elif isinstance(round_contrib, (list, np.ndarray)):
                    values = [v for v in round_contrib if v is not None and v > 0]
                
                if values:
                    contribution_data.append(values)
                    valid_rounds.append(rounds[i] if i < len(rounds) else i + 1)
            
            if contribution_data:
                bp = axes[1, 0].boxplot(contribution_data, positions=valid_rounds,
                                       widths=0.6, patch_artist=True, showfliers=False)
                
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                for median in bp['medians']:
                    median.set_color('red')
                    median.set_linewidth(2)
                
                means = [np.mean(data) for data in contribution_data]
                axes[1, 0].plot(valid_rounds, means, 'g--', linewidth=2, 
                              marker='D', markersize=5, label='Mean')
                
                axes[1, 0].set_xlabel('Round', fontsize=12)
                axes[1, 0].set_ylabel('AMAC Contribution', fontsize=12)
                axes[1, 0].set_title('Contribution Distribution', fontsize=13, fontweight='bold')
                axes[1, 0].legend(loc='best', fontsize=10)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                if valid_rounds:
                    axes[1, 0].set_xlim([min(valid_rounds) - 0.5, max(valid_rounds) + 0.5])
        
        # 时间消耗 / Time consumption
        if 'time_per_round' in metrics_history and metrics_history['time_per_round']:
            axes[1, 1].plot(rounds, metrics_history['time_per_round'], 
                          'g-', linewidth=2, marker='o', markersize=4)
            axes[1, 1].set_xlabel('Round', fontsize=12)
            axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
            axes[1, 1].set_title('Time per Round', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            avg_time = np.mean(metrics_history['time_per_round'])
            axes[1, 1].axhline(y=avg_time, color='r', linestyle='--', 
                             linewidth=1.5, label=f'Avg: {avg_time:.2f}s')
            axes[1, 1].legend(loc='best', fontsize=10)
        
        plt.suptitle(f'Training Progress - {experiment_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved: {save_path}")
    
    def plot_contribution_heatmap(self, contribution_matrix: np.ndarray,
                                 client_ids: List[int], rounds: List[int],
                                 experiment_name: str) -> None:
        """绘制贡献度热力图 / Plot contribution heatmap"""
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(contribution_matrix, 
                   xticklabels=rounds[::5],
                   yticklabels=client_ids,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'AMAC Contribution'})
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Client ID', fontsize=12)
        plt.title(f'Contribution Heatmap - {experiment_name}', fontsize=13)
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_contribution_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved: {save_path}")