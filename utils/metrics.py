"""
utils/metrics.py
评估指标模块（只关注客户端本地性能）/ Evaluation Metrics Module (Focus on Client Local Performance)
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime


class MetricsCalculator:
    """
    指标计算器 / Metrics Calculator
    专注于客户端本地测试集上的性能评估
    Focus on client performance evaluation on local test sets
    """
    
    def __init__(self):
        """初始化指标计算器 / Initialize metrics calculator"""
        # 性能指标 / Performance metrics
        self.round_metrics = []  # 存储每轮指标
        self.time_consumptions = []  # 时间消耗历史
        
        # 客户端准确率追踪 / Client accuracy tracking
        self.client_accuracies_per_round = []  # 每轮所有客户端的准确率
        self.avg_client_accuracies = []  # 每轮客户端平均准确率
        self.max_client_accuracies = []  # 每轮客户端最高准确率
        self.min_client_accuracies = []  # 每轮客户端最低准确率
        
        # PCC相关数据 / PCC related data
        self.standalone_accuracies = {}  # 客户端独立训练准确率（在本地测试集上）
        self.federated_accuracies = {}   # 客户端联邦学习准确率（在本地测试集上）
        
        # 贡献度数据 / Contribution data
        self.contribution_history = []
    
    def record_standalone_accuracy(self, client_id: int, accuracy: float) -> None:
        """
        记录客户端独立训练准确率（在本地测试集上）
        Record client standalone training accuracy (on local test set)
        
        Args:
            client_id: 客户端ID / Client ID
            accuracy: 独立训练准确率 / Standalone training accuracy
        """
        self.standalone_accuracies[client_id] = accuracy
    
    def record_federated_accuracy(self, client_id: int, accuracy: float) -> None:
        """
        记录客户端联邦学习准确率（在本地测试集上）
        Record client federated learning accuracy (on local test set)
        
        Args:
            client_id: 客户端ID / Client ID
            accuracy: 联邦学习准确率 / Federated learning accuracy
        """
        self.federated_accuracies[client_id] = accuracy
    
    def record_client_accuracies(self, client_accuracies: Dict[int, float]) -> None:
        """
        记录当前轮次所有客户端的准确率 / Record all clients' accuracies for current round
        
        Args:
            client_accuracies: 客户端ID到准确率的映射 / Mapping from client ID to accuracy
        """
        if client_accuracies:
            accuracies = list(client_accuracies.values())
            self.client_accuracies_per_round.append(client_accuracies.copy())
            self.avg_client_accuracies.append(np.mean(accuracies))
            self.max_client_accuracies.append(np.max(accuracies))
            self.min_client_accuracies.append(np.min(accuracies))
        else:
            self.client_accuracies_per_round.append({})
            self.avg_client_accuracies.append(0.0)
            self.max_client_accuracies.append(0.0)
            self.min_client_accuracies.append(0.0)
    
    def calculate_pcc(self) -> Tuple[float, Dict]:
        """
        计算皮尔逊相关系数 / Calculate Pearson Correlation Coefficient
        评估独立准确率与联邦学习准确率之间的相关性（均在本地测试集上）
        Evaluate correlation between standalone and federated accuracies (both on local test sets)
        
        Returns:
            (PCC值, 详细信息) / (PCC value, detailed information)
        """
        # 确保有数据 / Ensure data exists
        if not self.standalone_accuracies or not self.federated_accuracies:
            return 0.0, {'error': 'Insufficient data for PCC calculation'}
        
        # 获取共同的客户端ID / Get common client IDs
        common_clients = set(self.standalone_accuracies.keys()) & set(self.federated_accuracies.keys())
        
        if len(common_clients) < 2:
            return 0.0, {'error': 'Need at least 2 clients for PCC calculation'}
        
        # 准备数据向量 / Prepare data vectors
        standalone_vector = []
        federated_vector = []
        
        for client_id in sorted(common_clients):
            standalone_vector.append(self.standalone_accuracies[client_id])
            federated_vector.append(self.federated_accuracies[client_id])
        
        # 计算PCC / Calculate PCC
        pcc, p_value = stats.pearsonr(standalone_vector, federated_vector)
        
        # 准备详细信息 / Prepare detailed information
        details = {
            'pcc': pcc,
            'p_value': p_value,
            'num_clients': len(common_clients),
            'standalone_mean': np.mean(standalone_vector),
            'standalone_std': np.std(standalone_vector),
            'federated_mean': np.mean(federated_vector),
            'federated_std': np.std(federated_vector),
            'improvement_mean': np.mean([f - s for s, f in zip(standalone_vector, federated_vector)]),
            'improvement_std': np.std([f - s for s, f in zip(standalone_vector, federated_vector)]),
            'data_points': list(zip(standalone_vector, federated_vector))
        }
        
        return pcc, details
    
    def record_round(self, round_metrics: Dict):
        """
        记录单轮训练指标 / Record single round metrics
        
        Args:
            round_metrics: 包含轮次指标的字典，包括:
                - round: 轮次编号
                - time_consumption: 时间消耗
                - contributions: 客户端贡献度
                - client_accuracies: 客户端准确率
                等其他指标
        """
        # 保存完整的轮次指标
        self.round_metrics.append(round_metrics)
        
        # 提取关键指标到独立列表
        self.time_consumptions.append(round_metrics.get('time_consumption', 0))
        
        # 记录客户端准确率
        if 'client_accuracies' in round_metrics:
            self.record_client_accuracies(round_metrics['client_accuracies'])
        else:
            self.record_client_accuracies({})
    
    def calculate_final_metrics(self) -> Dict:
        """
        计算最终指标 / Calculate final metrics
        
        Returns:
            最终指标字典 / Final metrics dictionary
        """
        # 计算PCC / Calculate PCC
        pcc, pcc_details = self.calculate_pcc()
        
        # 计算客户端准确率统计
        client_acc_stats = {
            'avg_final': self.avg_client_accuracies[-1] if self.avg_client_accuracies else 0,
            'max_final': self.max_client_accuracies[-1] if self.max_client_accuracies else 0,
            'min_final': self.min_client_accuracies[-1] if self.min_client_accuracies else 0,
            'avg_mean': np.mean(self.avg_client_accuracies) if self.avg_client_accuracies else 0,
            'max_mean': np.mean(self.max_client_accuracies) if self.max_client_accuracies else 0,
            'min_mean': np.mean(self.min_client_accuracies) if self.min_client_accuracies else 0,
            'avg_improvement': pcc_details.get('improvement_mean', 0) if pcc_details else 0,
            'std_improvement': pcc_details.get('improvement_std', 0) if pcc_details else 0
        }
        
        # 计算时间消耗统计 / Calculate time consumption statistics
        time_stats = {
            'total': np.sum(self.time_consumptions) if self.time_consumptions else 0,
            'mean': np.mean(self.time_consumptions) if self.time_consumptions else 0,
            'max': np.max(self.time_consumptions) if self.time_consumptions else 0,
            'min': np.min(self.time_consumptions) if self.time_consumptions else 0
        }
        
        return {
            'client_accuracy': client_acc_stats,
            'time_consumption': time_stats,
            'pcc': pcc,
            'pcc_details': pcc_details,
            'num_rounds': len(self.round_metrics),
            'num_clients': len(self.standalone_accuracies)
        }
    
    def get_pcc_visualization_data(self) -> Dict:
        """
        获取PCC可视化数据 / Get PCC visualization data
        
        Returns:
            可视化数据字典 / Visualization data dictionary
        """
        if not self.standalone_accuracies or not self.federated_accuracies:
            return {}
        
        common_clients = set(self.standalone_accuracies.keys()) & set(self.federated_accuracies.keys())
        
        data = {
            'client_ids': list(common_clients),
            'standalone_accuracies': [self.standalone_accuracies[cid] for cid in common_clients],
            'federated_accuracies': [self.federated_accuracies[cid] for cid in common_clients],
            'improvements': [
                self.federated_accuracies[cid] - self.standalone_accuracies[cid] 
                for cid in common_clients
            ]
        }
        
        return data
    
    def save_metrics(self, filepath: str) -> None:
        """
        保存指标到文件 / Save metrics to file
        
        Args:
            filepath: 文件路径 / File path
        """
        metrics = self.calculate_final_metrics()
        
        # 添加原始数据 / Add raw data
        metrics['raw_data'] = {
            'round_metrics': self.round_metrics,
            'standalone_accuracies': self.standalone_accuracies,
            'federated_accuracies': self.federated_accuracies,
            'avg_client_accuracies': self.avg_client_accuracies,
            'max_client_accuracies': self.max_client_accuracies,
            'min_client_accuracies': self.min_client_accuracies
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Metrics saved to: {filepath}")
    
    def print_summary(self) -> None:
        """
        打印指标摘要 / Print metrics summary
        """
        metrics = self.calculate_final_metrics()
        
        print("\n" + "="*70)
        print("Performance Metrics Summary / 性能指标摘要")
        print("Note: All accuracies measured on clients' local test sets")
        print("注意：所有准确率均在客户端本地测试集上测量")
        print("="*70)
        
        print("\n1. Client Accuracy (on local test sets) / 客户端准确率（本地测试集）:")
        print(f"   Avg Client Final: {metrics['client_accuracy']['avg_final']:.4f}")
        print(f"   Max Client Final: {metrics['client_accuracy']['max_final']:.4f}")
        print(f"   Min Client Final: {metrics['client_accuracy']['min_final']:.4f}")
        print(f"   Avg Client Mean:  {metrics['client_accuracy']['avg_mean']:.4f}")
        
        print("\n2. Performance Improvement / 性能提升:")
        print(f"   Avg Improvement: {metrics['client_accuracy']['avg_improvement']:.4f} ({metrics['client_accuracy']['avg_improvement']*100:.2f}%)")
        print(f"   Std Improvement: {metrics['client_accuracy']['std_improvement']:.4f}")
        
        print("\n3. Time Consumption / 时间消耗:")
        print(f"   Total: {metrics['time_consumption']['total']:.2f} seconds")
        print(f"   Mean:  {metrics['time_consumption']['mean']:.2f} seconds/round")
        
        print("\n4. Pearson Correlation Coefficient / 皮尔逊相关系数:")
        print(f"   PCC:   {metrics['pcc']:.4f}")
        if 'pcc_details' in metrics and 'p_value' in metrics['pcc_details']:
            print(f"   p-value: {metrics['pcc_details']['p_value']:.4f}")
            print(f"   Interpretation: ", end="")
            
            pcc = metrics['pcc']
            if abs(pcc) >= 0.8:
                print("Strong correlation / 强相关")
            elif abs(pcc) >= 0.5:
                print("Moderate correlation / 中等相关")
            elif abs(pcc) >= 0.3:
                print("Weak correlation / 弱相关")
            else:
                print("Very weak or no correlation / 极弱相关或无相关")
        
        print("="*70)