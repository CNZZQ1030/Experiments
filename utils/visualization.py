"""
utils/visualization.py
可视化模块（包含PCC）/ Visualization Module (including PCC)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import os


class Visualizer:
    """
    可视化工具 / Visualization Tool
    包含PCC散点图和相关性分析可视化
    Including PCC scatter plot and correlation analysis visualization
    """
    
    def __init__(self, save_dir: str = "outputs/plots"):
        """
        初始化可视化器 / Initialize visualizer
        
        Args:
            save_dir: 图表保存目录 / Chart save directory
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
        显示独立准确率与联邦学习准确率的相关性
        Show correlation between standalone and federated accuracies
        
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
        
        ax1.set_xlabel('Standalone Accuracy / 独立训练准确率', fontsize=12)
        ax1.set_ylabel('Federated Accuracy / 联邦学习准确率', fontsize=12)
        ax1.set_title(f'Correlation Analysis / 相关性分析\nPCC = {pcc_value:.4f}, p-value = {p_value:.4f}', 
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
        
        ax2.set_xlabel('Accuracy Improvement / 准确率提升', fontsize=12)
        ax2.set_ylabel('Number of Clients / 客户端数量', fontsize=12)
        ax2.set_title('Distribution of Performance Improvement / 性能提升分布', fontsize=13)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'PCC Analysis - {experiment_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_pcc_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCC scatter plot saved to: {save_path}")
    
    def plot_training_curves(self, metrics_history: Dict, experiment_name: str) -> None:
        """
        绘制训练曲线 / Plot training curves
        
        Args:
            metrics_history: 指标历史 / Metrics history
            experiment_name: 实验名称 / Experiment name
        """
        rounds = metrics_history.get('rounds', list(range(len(metrics_history['accuracy']))))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 准确率曲线 / Accuracy curve
        axes[0, 0].plot(rounds, metrics_history['accuracy'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Round / 轮次')
        axes[0, 0].set_ylabel('Accuracy / 准确率')
        axes[0, 0].set_title('Test Accuracy over Training / 训练过程中的测试准确率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 损失曲线 / Loss curve
        axes[0, 1].plot(rounds, metrics_history['loss'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Round / 轮次')
        axes[0, 1].set_ylabel('Loss / 损失')
        axes[0, 1].set_title('Training Loss / 训练损失')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 贡献度分布 / Contribution distribution
        if 'contributions' in metrics_history:
            contributions = metrics_history['contributions']
            axes[1, 0].boxplot([c for c in contributions], positions=rounds)
            axes[1, 0].set_xlabel('Round / 轮次')
            axes[1, 0].set_ylabel('AMAC Contribution / AMAC贡献度')
            axes[1, 0].set_title('Client Contribution Distribution / 客户端贡献度分布')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 时间消耗 / Time consumption
        if 'time_per_round' in metrics_history:
            axes[1, 1].plot(rounds, metrics_history['time_per_round'], 'g-', linewidth=2)
            axes[1, 1].set_xlabel('Round / 轮次')
            axes[1, 1].set_ylabel('Time (seconds) / 时间 (秒)')
            axes[1, 1].set_title('Time Consumption per Round / 每轮时间消耗')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Progress - {experiment_name}', fontsize=14, fontweight='bold')
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
                   cbar_kws={'label': 'AMAC Contribution / AMAC贡献度'})
        
        plt.xlabel('Round / 轮次', fontsize=12)
        plt.ylabel('Client ID / 客户端ID', fontsize=12)
        plt.title(f'Client Contribution Heatmap / 客户端贡献度热力图\n{experiment_name}', 
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
        ax1.set_xlabel('Client ID / 客户端ID')
        ax1.set_ylabel('Model Quality (Accuracy) / 模型质量 (准确率)')
        ax1.set_title('Final Model Quality by Client / 客户端最终模型质量')
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
                ax2.plot(rounds, client_qualities[cid], 
                        label=f'Client {cid}', linewidth=2)
        
        ax2.set_xlabel('Round / 轮次')
        ax2.set_ylabel('Model Quality / 模型质量')
        ax2.set_title('Model Quality Evolution / 模型质量演变')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Quality Analysis - {experiment_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'{experiment_name}_model_quality.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model quality comparison saved to: {save_path}")