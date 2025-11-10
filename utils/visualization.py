"""
utils/visualization.py
可视化模块（专注于客户端本地性能）/ Visualization Module (Focus on Client Local Performance)
"""
from typing import Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """
    Visualization Tool
    专注于客户端本地测试集上的性能可视化
    Focus on visualizing client performance on local test sets
    """
    
    def __init__(self, save_dir: str = "outputs/plots"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Chart save directory
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置绘图风格 / Set plotting style
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
        显示独立准确率与联邦学习准确率的相关性（均在本地测试集上）
        Show correlation between standalone and federated accuracies (both on local test sets)
        
        Args:
            standalone_accuracies: 独立训练准确率列表 / Standalone training accuracy list
            federated_accuracies: 联邦学习准确率列表 / Federated learning accuracy list
            pcc_value: PCC值 / PCC value
            p_value: p值 / p-value
            experiment_name: 实验名称 / Experiment name
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：散点图与回归线 / Left: Scatter plot with regression line
        ax1.scatter(standalone_accuracies, federated_accuracies, 
                   alpha=0.6, s=50, c='blue', edgecolors='navy', linewidth=1.5)
        
        # 添加回归线 / Add regression line
        z = np.polyfit(standalone_accuracies, federated_accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(standalone_accuracies), max(standalone_accuracies), 100)
        ax1.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2, label=f'Regression line')
        
        # 添加对角线（完美相关）/ Add diagonal line (perfect correlation)
        min_val = min(min(standalone_accuracies), min(federated_accuracies))
        max_val = max(max(standalone_accuracies), max(federated_accuracies))
        ax1.plot([min_val, max_val], [min_val, max_val], 'g--', 
                alpha=0.5, linewidth=1.5, label='Perfect correlation')
        
        ax1.set_xlabel('Standalone Accuracy (Local Test)', fontsize=12)
        ax1.set_ylabel('Federated Accuracy (Local Test)', fontsize=12)
        ax1.set_title(f'Correlation Analysis\nPCC = {pcc_value:.4f}, p-value = {p_value:.4f}', 
                     fontsize=13)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 设置坐标轴范围 / Set axis range
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
        ax2.set_title('Distribution of Performance Improvement', fontsize=13)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'PCC Analysis - {experiment_name}\n(Measured on clients\' local test sets)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_pcc_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCC scatter plot saved to: {save_path}")
    
    def plot_training_curves(self, metrics_history: Dict, experiment_name: str) -> None:
        """
        绘制训练曲线 / Plot training curves
        
        Args:
            metrics_history: Metrics history containing:
                - rounds: 轮次列表 / Round list
                - avg_client_accuracy: 客户端平均准确率 / Average client accuracy
                - max_client_accuracy: 客户端最高准确率 / Max client accuracy
                - time_per_round: 每轮时间 / Time per round
                - contributions: 贡献度 / Contributions
            experiment_name: Experiment name
        """
        rounds = metrics_history.get('rounds', list(range(1, len(metrics_history.get('avg_client_accuracy', [])) + 1)))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ========== 准确率曲线（双线：平均和最高）/ Accuracy curve (two lines: avg and max) ==========
        if 'avg_client_accuracy' in metrics_history and metrics_history['avg_client_accuracy']:
            axes[0, 0].plot(rounds, metrics_history['avg_client_accuracy'], 
                          'b-', linewidth=2.5, label='Avg Client', marker='o', markersize=5)
        
        if 'max_client_accuracy' in metrics_history and metrics_history['max_client_accuracy']:
            axes[0, 0].plot(rounds, metrics_history['max_client_accuracy'], 
                          'r-', linewidth=2.5, label='Max Client', marker='^', markersize=5)
        
        axes[0, 0].set_xlabel('Round', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy (on local test sets)', fontsize=12)
        axes[0, 0].set_title('Client Accuracy over Training\n(Measured on local test sets)', 
                            fontsize=13, fontweight='bold')
        axes[0, 0].legend(loc='best', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加y轴范围提示
        if metrics_history.get('avg_client_accuracy'):
            y_min = min(metrics_history['avg_client_accuracy'])
            y_max = max(metrics_history.get('max_client_accuracy', metrics_history['avg_client_accuracy']))
            axes[0, 0].set_ylim([max(0, y_min - 0.05), min(1.0, y_max + 0.05)])
        
        # ========== 准确率差异（最高-平均）/ Accuracy gap (max - avg) ==========
        if 'avg_client_accuracy' in metrics_history and 'max_client_accuracy' in metrics_history:
            if metrics_history['avg_client_accuracy'] and metrics_history['max_client_accuracy']:
                accuracy_gap = [max_acc - avg_acc for max_acc, avg_acc in 
                              zip(metrics_history['max_client_accuracy'], 
                                  metrics_history['avg_client_accuracy'])]
                
                axes[0, 1].plot(rounds, accuracy_gap, 'purple', linewidth=2, marker='s', markersize=4)
                axes[0, 1].set_xlabel('Round', fontsize=12)
                axes[0, 1].set_ylabel('Accuracy Gap', fontsize=12)
                axes[0, 1].set_title('Client Performance Gap\n(Max - Avg)', fontsize=13, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 添加趋势线
                if len(accuracy_gap) > 1:
                    z = np.polyfit(rounds, accuracy_gap, 1)
                    p = np.poly1d(z)
                    axes[0, 1].plot(rounds, p(rounds), "r--", alpha=0.5, linewidth=1.5, 
                                  label=f'Trend: {"↓" if z[0] < 0 else "↑"}')
                    axes[0, 1].legend(loc='best')
        
        # ========== 贡献度分布 / Contribution distribution ==========
        if 'contributions' in metrics_history and metrics_history['contributions']:
            contributions = metrics_history['contributions']
            
            # 准备箱线图数据 / Prepare box plot data
            contribution_data = []
            valid_rounds = []
            
            for i, round_contributions in enumerate(contributions):
                values = []
                
                # 处理不同格式的贡献度数据 / Handle different formats of contribution data
                if isinstance(round_contributions, dict):
                    values = [v for v in round_contributions.values() if v is not None and v > 0]
                elif isinstance(round_contributions, (list, np.ndarray)):
                    values = [v for v in round_contributions if v is not None and v > 0]
                
                if values:  # 只添加非空数据 / Only add non-empty data
                    contribution_data.append(values)
                    valid_rounds.append(rounds[i] if i < len(rounds) else i + 1)
            
            if contribution_data and len(contribution_data) > 0:
                # 绘制箱线图 / Draw box plot
                bp = axes[1, 0].boxplot(contribution_data, positions=valid_rounds,
                                       widths=0.6, patch_artist=True,
                                       showfliers=False)  # 不显示异常值
                
                # 美化箱线图 / Beautify box plot
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                for median in bp['medians']:
                    median.set_color('red')
                    median.set_linewidth(2)
                
                # 添加平均值线 / Add mean line
                means = [np.mean(data) for data in contribution_data]
                axes[1, 0].plot(valid_rounds, means, 'g--', linewidth=2, 
                              marker='D', markersize=5, label='Mean')
                
                axes[1, 0].set_xlabel('Round', fontsize=12)
                axes[1, 0].set_ylabel('AMAC Contribution', fontsize=12)
                axes[1, 0].set_title('Client Contribution Distribution', fontsize=13, fontweight='bold')
                axes[1, 0].legend(loc='best', fontsize=10)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # 设置x轴范围 / Set x-axis range
                if valid_rounds:
                    axes[1, 0].set_xlim([min(valid_rounds) - 0.5, max(valid_rounds) + 0.5])
            else:
                # 如果没有有效数据 / If no valid data
                axes[1, 0].text(0.5, 0.5, 'No valid contribution data\n(contributions may all be 0 or None)', 
                              ha='center', va='center', fontsize=12, color='gray',
                              transform=axes[1, 0].transAxes)
                axes[1, 0].set_xlabel('Round', fontsize=12)
                axes[1, 0].set_ylabel('AMAC Contribution', fontsize=12)
                axes[1, 0].set_title('Client Contribution Distribution', fontsize=13, fontweight='bold')
        else:
            # 如果完全没有贡献度数据 / If no contribution data at all
            axes[1, 0].text(0.5, 0.5, 'No contribution data available', 
                          ha='center', va='center', fontsize=12, color='gray',
                          transform=axes[1, 0].transAxes)
            axes[1, 0].set_xlabel('Round', fontsize=12)
            axes[1, 0].set_ylabel('AMAC Contribution', fontsize=12)
            axes[1, 0].set_title('Client Contribution Distribution', fontsize=13, fontweight='bold')
        
        # ========== 时间消耗 / Time consumption ==========
        if 'time_per_round' in metrics_history and metrics_history['time_per_round']:
            axes[1, 1].plot(rounds, metrics_history['time_per_round'], 
                          'g-', linewidth=2, marker='o', markersize=4)
            axes[1, 1].set_xlabel('Round', fontsize=12)
            axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
            axes[1, 1].set_title('Time Consumption per Round', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加平均时间线 / Add average time line
            avg_time = np.mean(metrics_history['time_per_round'])
            axes[1, 1].axhline(y=avg_time, color='r', linestyle='--', 
                             linewidth=1.5, label=f'Avg: {avg_time:.2f}s')
            axes[1, 1].legend(loc='best', fontsize=10)
        
        plt.suptitle(f'Training Progress - {experiment_name}\n(Clients evaluated on their own local test sets)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {save_path}")
    
    def plot_contribution_heatmap(self, contribution_matrix: np.ndarray,
                                 client_ids: List[int], rounds: List[int],
                                 experiment_name: str) -> None:
        """
        绘制贡献度热力图 / Plot contribution heatmap
        
        Args:
            contribution_matrix: 贡献度矩阵 (clients x rounds) / Contribution matrix
            client_ids: 客户端ID列表 / Client ID list
            rounds: 轮次列表 / Round list
            experiment_name: 实验名称 / Experiment name
        """
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(contribution_matrix, 
                   xticklabels=rounds[::5],  # 每5轮显示一个标签
                   yticklabels=client_ids,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'AMAC Contribution'})
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Client ID', fontsize=12)
        plt.title(f'Client Contribution Heatmap - {experiment_name}', 
                 fontsize=13)
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_contribution_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Contribution heatmap saved to: {save_path}")
    
    def plot_model_quality_comparison(self, client_qualities: Dict[int, List[float]],
                                     experiment_name: str) -> None:
        """
        绘制模型质量对比 / Plot model quality comparison
        比较不同贡献度客户端获得的模型质量
        Compare model quality received by clients with different contributions
        
        Args:
            client_qualities: 客户端模型质量字典 / Client model quality dictionary
            experiment_name: 实验名称 / Experiment name
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按最终质量排序客户端 / Sort clients by final quality
        final_qualities = {cid: qualities[-1] for cid, qualities in client_qualities.items()}
        sorted_clients = sorted(final_qualities.items(), key=lambda x: x[1], reverse=True)
        
        # 左图：客户端模型质量条形图 / Left: Client model quality bar chart
        client_ids = [str(cid) for cid, _ in sorted_clients[:20]]  # 显示前20个
        qualities = [q for _, q in sorted_clients[:20]]
        
        bars = ax1.bar(range(len(client_ids)), qualities, color='steelblue')
        
        # 根据质量着色 / Color by quality
        for i, bar in enumerate(bars):
            if qualities[i] >= 0.8:
                bar.set_color('green')
            elif qualities[i] >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax1.set_xticks(range(len(client_ids)))
        ax1.set_xticklabels(client_ids, rotation=45)
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Model Quality (Accuracy on Local Test)')
        ax1.set_title('Final Model Quality by Client')
        ax1.grid(True, alpha=0.3)
        
        # 右图：质量演变 / Right: Quality evolution
        rounds = list(range(len(next(iter(client_qualities.values())))))
        
        # 选择几个代表性客户端 / Select representative clients
        percentiles = [10, 25, 50, 75, 90]
        selected_clients = []
        for p in percentiles:
            idx = int(len(sorted_clients) * p / 100)
            if idx < len(sorted_clients):
                selected_clients.append(sorted_clients[idx][0])
        
        for cid in selected_clients:
            if cid in client_qualities:
                ax2.plot(rounds, client_qualities[cid], label=f'Client {cid}', linewidth=2)
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Model Quality (Local Test)')
        ax2.set_title('Model Quality Evolution')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Quality Analysis - {experiment_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_model_quality.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model quality comparison saved to: {save_path}")