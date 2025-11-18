"""
utils/metrics.py (Updated with IPR)
评估指标模块 - 增加IPR指标 / Evaluation Metrics Module - Added IPR metrics
专注于客户端在各自测试集上的性能评估
Focus on client performance on their own test sets
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
    评估客户端在各自测试集上的性能，包括IPR指标
    Evaluate client performance on their own test sets, including IPR metrics
    """
    
    def __init__(self):
        """初始化 / Initialize"""
        # 性能指标 / Performance metrics
        self.round_metrics = []
        self.time_consumptions = []
        
        # 客户端准确率 / Client accuracies
        self.client_accuracies_per_round = []
        self.avg_client_accuracies = []
        self.max_client_accuracies = []
        self.min_client_accuracies = []
        
        # PCC数据 / PCC data
        self.standalone_accuracies = {}  # 独立训练准确率
        self.federated_accuracies = {}   # 联邦学习准确率
        
        # 贡献度 / Contributions
        self.contribution_history = []
        
        # IPR相关 / IPR related
        self.ipr_per_round = []  # 每轮的IPR
        
    def record_standalone_accuracy(self, client_id: int, accuracy: float) -> None:
        """记录独立训练准确率 / Record standalone accuracy"""
        self.standalone_accuracies[client_id] = accuracy
    
    def record_federated_accuracy(self, client_id: int, accuracy: float) -> None:
        """记录联邦学习准确率 / Record federated accuracy"""
        self.federated_accuracies[client_id] = accuracy
    
    def record_client_accuracies(self, client_accuracies: Dict[int, float]) -> None:
        """记录当前轮次所有客户端准确率 / Record all client accuracies"""
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
    
    def calculate_ipr_accuracy(self, current_round: int = None) -> Tuple[float, Dict]:
        """
        计算基于准确率的IPR / Calculate IPR based on accuracy
        
        IPR_accu = (满足 Perf_FL,i(accuracy) ≥ Perf_standalone,i(accuracy) 的客户端数量) / N
        
        Args:
            current_round: 当前轮次，如果为None则使用最终准确率 / Current round
            
        Returns:
            (IPR值, 详细信息) / (IPR value, details)
        """
        if not self.standalone_accuracies or not self.federated_accuracies:
            return 0.0, {'error': 'Insufficient data'}
        
        common_clients = set(self.standalone_accuracies.keys()) & set(self.federated_accuracies.keys())
        
        if len(common_clients) == 0:
            return 0.0, {'error': 'No common clients'}
        
        # 计算获益的客户端数量 / Count clients that benefited
        benefited_clients = 0
        client_improvements = {}
        
        for client_id in common_clients:
            standalone_acc = self.standalone_accuracies[client_id]
            federated_acc = self.federated_accuracies[client_id]
            
            # 联邦学习准确率 >= 独立训练准确率
            if federated_acc >= standalone_acc:
                benefited_clients += 1
            
            improvement = federated_acc - standalone_acc
            client_improvements[client_id] = {
                'standalone': standalone_acc,
                'federated': federated_acc,
                'improvement': improvement,
                'benefited': improvement >= 0
            }
        
        # 计算IPR / Calculate IPR
        ipr_accuracy = benefited_clients / len(common_clients)
        
        details = {
            'ipr_accuracy': ipr_accuracy,
            'num_benefited': benefited_clients,
            'total_clients': len(common_clients),
            'benefited_percentage': ipr_accuracy * 100,
            'client_improvements': client_improvements,
            'avg_improvement': np.mean([info['improvement'] for info in client_improvements.values()]),
            'std_improvement': np.std([info['improvement'] for info in client_improvements.values()]),
            'min_improvement': np.min([info['improvement'] for info in client_improvements.values()]),
            'max_improvement': np.max([info['improvement'] for info in client_improvements.values()])
        }
        
        return ipr_accuracy, details
    
    def calculate_ipr_per_round(self, round_num: int, 
                                current_accuracies: Dict[int, float]) -> float:
        """
        计算当前轮次的IPR / Calculate IPR for current round
        
        Args:
            round_num: 轮次编号 / Round number
            current_accuracies: 当前轮次的准确率 / Current round accuracies
            
        Returns:
            当前轮次的IPR / IPR for current round
        """
        if not self.standalone_accuracies or not current_accuracies:
            return 0.0
        
        common_clients = set(self.standalone_accuracies.keys()) & set(current_accuracies.keys())
        
        if len(common_clients) == 0:
            return 0.0
        
        benefited = sum(1 for cid in common_clients 
                       if current_accuracies[cid] >= self.standalone_accuracies[cid])
        
        ipr = benefited / len(common_clients)
        return ipr
    
    def calculate_pcc(self) -> Tuple[float, Dict]:
        """
        计算PCC / Calculate Pearson Correlation Coefficient
        评估独立训练与联邦学习准确率的相关性
        Evaluate correlation between standalone and federated accuracies
        
        Returns:
            (PCC值, 详细信息) / (PCC value, details)
        """
        if not self.standalone_accuracies or not self.federated_accuracies:
            return 0.0, {'error': 'Insufficient data'}
        
        common_clients = set(self.standalone_accuracies.keys()) & set(self.federated_accuracies.keys())
        
        if len(common_clients) < 2:
            return 0.0, {'error': 'Need at least 2 clients'}
        
        standalone_vector = []
        federated_vector = []
        
        for client_id in sorted(common_clients):
            standalone_vector.append(self.standalone_accuracies[client_id])
            federated_vector.append(self.federated_accuracies[client_id])
        
        pcc, p_value = stats.pearsonr(standalone_vector, federated_vector)
        
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
        记录轮次指标 / Record round metrics
        
        Args:
            round_metrics: 轮次指标字典 / Round metrics dictionary
        """
        self.round_metrics.append(round_metrics)
        self.time_consumptions.append(round_metrics.get('time_consumption', 0))
        
        if 'client_accuracies' in round_metrics:
            # 记录准确率
            self.record_client_accuracies(round_metrics['client_accuracies'])
            
            # 计算当前轮次的IPR
            current_ipr = self.calculate_ipr_per_round(
                round_metrics['round'],
                round_metrics['client_accuracies']
            )
            self.ipr_per_round.append(current_ipr)
        else:
            self.record_client_accuracies({})
            self.ipr_per_round.append(0.0)
    
    def calculate_final_metrics(self) -> Dict:
        """计算最终指标 / Calculate final metrics"""
        pcc, pcc_details = self.calculate_pcc()
        ipr_accuracy, ipr_details = self.calculate_ipr_accuracy()
        
        client_acc_stats = {
            'avg_final': self.avg_client_accuracies[-1] if self.avg_client_accuracies else 0,
            'max_final': self.max_client_accuracies[-1] if self.max_client_accuracies else 0,
            'min_final': self.min_client_accuracies[-1] if self.min_client_accuracies else 0,
            'avg_mean': np.mean(self.avg_client_accuracies) if self.avg_client_accuracies else 0,
            'max_mean': np.mean(self.max_client_accuracies) if self.max_client_accuracies else 0,
            'min_mean': np.mean(self.min_client_accuracies) if self.min_client_accuracies else 0,
            'avg_improvement': ipr_details.get('avg_improvement', 0) if ipr_details else 0,
            'std_improvement': ipr_details.get('std_improvement', 0) if ipr_details else 0
        }
        
        time_stats = {
            'total': np.sum(self.time_consumptions) if self.time_consumptions else 0,
            'mean': np.mean(self.time_consumptions) if self.time_consumptions else 0,
            'max': np.max(self.time_consumptions) if self.time_consumptions else 0,
            'min': np.min(self.time_consumptions) if self.time_consumptions else 0
        }
        
        # IPR统计 / IPR statistics
        ipr_stats = {
            'final_ipr': ipr_accuracy,
            'ipr_percentage': ipr_accuracy * 100,
            'num_benefited': ipr_details.get('num_benefited', 0) if ipr_details else 0,
            'ipr_history': self.ipr_per_round,
            'avg_ipr': np.mean(self.ipr_per_round) if self.ipr_per_round else 0,
            'final_10_rounds_avg_ipr': np.mean(self.ipr_per_round[-10:]) if len(self.ipr_per_round) >= 10 else np.mean(self.ipr_per_round) if self.ipr_per_round else 0
        }
        
        return {
            'client_accuracy': client_acc_stats,
            'time_consumption': time_stats,
            'pcc': pcc,
            'pcc_details': pcc_details,
            'ipr': ipr_stats,
            'ipr_details': ipr_details,
            'num_rounds': len(self.round_metrics),
            'num_clients': len(self.standalone_accuracies)
        }
    
    def get_pcc_visualization_data(self) -> Dict:
        """获取PCC可视化数据 / Get PCC visualization data"""
        if not self.standalone_accuracies or not self.federated_accuracies:
            return {}
        
        common_clients = set(self.standalone_accuracies.keys()) & set(self.federated_accuracies.keys())
        
        return {
            'client_ids': list(common_clients),
            'standalone_accuracies': [self.standalone_accuracies[cid] for cid in common_clients],
            'federated_accuracies': [self.federated_accuracies[cid] for cid in common_clients],
            'improvements': [
                self.federated_accuracies[cid] - self.standalone_accuracies[cid] 
                for cid in common_clients
            ]
        }
    
    def get_ipr_visualization_data(self) -> Dict:
        """
        获取IPR可视化数据 / Get IPR visualization data
        
        Returns:
            IPR可视化所需的数据 / Data needed for IPR visualization
        """
        ipr_accuracy, ipr_details = self.calculate_ipr_accuracy()
        
        if 'client_improvements' not in ipr_details:
            return {}
        
        client_improvements = ipr_details['client_improvements']
        
        # 按改进程度排序 / Sort by improvement
        sorted_clients = sorted(client_improvements.items(), 
                              key=lambda x: x[1]['improvement'],
                              reverse=True)
        
        return {
            'client_ids': [cid for cid, _ in sorted_clients],
            'improvements': [info['improvement'] for _, info in sorted_clients],
            'benefited': [info['benefited'] for _, info in sorted_clients],
            'standalone_accuracies': [info['standalone'] for _, info in sorted_clients],
            'federated_accuracies': [info['federated'] for _, info in sorted_clients],
            'ipr_value': ipr_accuracy,
            'ipr_percentage': ipr_accuracy * 100,
            'num_benefited': ipr_details['num_benefited'],
            'total_clients': ipr_details['total_clients'],
            'ipr_history': self.ipr_per_round
        }
    
    def save_metrics(self, filepath: str) -> None:
        """保存指标 / Save metrics"""
        metrics = self.calculate_final_metrics()
        
        metrics['raw_data'] = {
            'round_metrics': self.round_metrics,
            'standalone_accuracies': self.standalone_accuracies,
            'federated_accuracies': self.federated_accuracies,
            'avg_client_accuracies': self.avg_client_accuracies,
            'max_client_accuracies': self.max_client_accuracies,
            'min_client_accuracies': self.min_client_accuracies,
            'ipr_per_round': self.ipr_per_round
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Metrics saved to: {filepath}")
    
    def print_summary(self) -> None:
        """打印摘要 / Print summary"""
        metrics = self.calculate_final_metrics()
        
        print("\n" + "="*70)
        print("Performance Summary / 性能摘要")
        print("Note: All accuracies measured on clients' own test sets")
        print("注意：所有准确率均在客户端各自的测试集上测量")
        print("="*70)
        
        print("\n1. Client Accuracy / 客户端准确率:")
        print(f"   Avg Final: {metrics['client_accuracy']['avg_final']:.4f}")
        print(f"   Max Final: {metrics['client_accuracy']['max_final']:.4f}")
        print(f"   Min Final: {metrics['client_accuracy']['min_final']:.4f}")
        print(f"   Avg Mean:  {metrics['client_accuracy']['avg_mean']:.4f}")
        
        print("\n2. Performance Improvement / 性能提升:")
        print(f"   Avg Improvement: {metrics['client_accuracy']['avg_improvement']:.4f} "
              f"({metrics['client_accuracy']['avg_improvement']*100:.2f}%)")
        print(f"   Std Improvement: {metrics['client_accuracy']['std_improvement']:.4f}")
        
        print("\n3. IPR (Incentivized Participation Rate) / 激励参与率:")
        print(f"   Final IPR: {metrics['ipr']['final_ipr']:.4f} ({metrics['ipr']['ipr_percentage']:.2f}%)")
        print(f"   Benefited Clients: {metrics['ipr']['num_benefited']}/{metrics['num_clients']}")
        print(f"   Avg IPR (all rounds): {metrics['ipr']['avg_ipr']:.4f}")
        print(f"   Avg IPR (last 10 rounds): {metrics['ipr']['final_10_rounds_avg_ipr']:.4f}")
        
        ipr_value = metrics['ipr']['final_ipr']
        print(f"   Interpretation: ", end="")
        if ipr_value >= 0.95:
            print("Excellent! Nearly all clients benefit / 极好！几乎所有客户端受益")
        elif ipr_value >= 0.80:
            print("Good! Most clients benefit / 良好！大多数客户端受益")
        elif ipr_value >= 0.60:
            print("Moderate. Many clients benefit / 中等。较多客户端受益")
        else:
            print("Low. Consider improving incentive mechanism / 较低。考虑改进激励机制")
        
        print("\n4. Time Consumption / 时间消耗:")
        print(f"   Total: {metrics['time_consumption']['total']:.2f} seconds")
        print(f"   Mean:  {metrics['time_consumption']['mean']:.2f} seconds/round")
        
        print("\n5. Pearson Correlation Coefficient / 皮尔逊相关系数:")
        print(f"   PCC: {metrics['pcc']:.4f}")
        if 'pcc_details' in metrics and 'p_value' in metrics['pcc_details']:
            print(f"   p-value: {metrics['pcc_details']['p_value']:.4f}")
            
            pcc = metrics['pcc']
            print(f"   Interpretation: ", end="")
            if abs(pcc) >= 0.8:
                print("Strong correlation / 强相关")
            elif abs(pcc) >= 0.5:
                print("Moderate correlation / 中等相关")
            elif abs(pcc) >= 0.3:
                print("Weak correlation / 弱相关")
            else:
                print("Very weak or no correlation / 极弱相关或无相关")
        
        print("="*70)