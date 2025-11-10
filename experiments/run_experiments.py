"""
experiments/run_experiments.py
实验运行器 / Experiment Runner
专注于客户端个性化模型性能评估
Focus on client personalized model performance evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
import json
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.data_loader import FederatedDataLoader
from models.cnn_model import ModelFactory
from federated.server import FederatedServer
from federated.client import FederatedClient
from incentive.time_slice import TimeSliceManager
from incentive.membership import MembershipSystem
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer
from config import FederatedConfig, IncentiveConfig, DatasetConfig, DEVICE


class ExperimentRunner:
    """
    实验运行器 / Experiment Runner
    评估联邦学习激励机制对客户端性能的影响
    Evaluate impact of federated learning incentive mechanism on client performance
    """
    
    def __init__(self, dataset_name: str, num_clients: int, num_rounds: int,
                 distribution: str = "iid", time_slice_type: str = "rounds",
                 device: torch.device = None):
        """
        初始化实验运行器 / Initialize experiment runner
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            num_clients: 客户端数量 / Number of clients
            num_rounds: 训练轮次 / Training rounds
            distribution: 数据分布 (iid/non-iid) / Data distribution
            time_slice_type: 时间片类型 / Time slice type
            device: 计算设备 / Computing device
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.distribution = distribution
        self.time_slice_type = time_slice_type
        self.device = device or DEVICE
        
        print(f"\n{'='*70}")
        print(f"Experiment Initialization / 实验初始化")
        print(f"{'='*70}")
        print(f"Dataset / 数据集: {dataset_name}")
        print(f"Clients / 客户端数: {num_clients}")
        print(f"Rounds / 轮次: {num_rounds}")
        print(f"Distribution / 分布: {distribution}")
        print(f"Time Slice / 时间片: {time_slice_type}")
        print(f"Device / 设备: {self.device}")
        print(f"\nGoal / 目标: Compare standalone vs federated learning performance")
        print(f"目标: 比较独立训练与联邦学习的性能差异")
        
        # 初始化组件 / Initialize components
        self._initialize_data()
        self._initialize_model()
        self._initialize_server()
        self._initialize_clients()
        self._initialize_incentive_system()
        self._initialize_metrics()
        
    def _initialize_data(self):
        """初始化数据 / Initialize data"""
        self.data_loader = FederatedDataLoader(
            dataset_name=self.dataset_name,
            num_clients=self.num_clients,
            batch_size=FederatedConfig.LOCAL_BATCH_SIZE,
            distribution=self.distribution,
            alpha=FederatedConfig.NON_IID_ALPHA
        )
    
    def _initialize_model(self):
        """初始化模型 / Initialize model"""
        num_classes = DatasetConfig.NUM_CLASSES[self.dataset_name]
        input_channels = DatasetConfig.INPUT_SHAPE[self.dataset_name][0]
        
        self.model = ModelFactory.create_model(
            self.dataset_name,
            num_classes=num_classes,
            input_channels=input_channels
        )
    
    def _initialize_server(self):
        """初始化服务器 / Initialize server"""
        self.server = FederatedServer(self.model, self.device)
    
    def _initialize_clients(self):
        """初始化客户端 / Initialize clients"""
        self.clients = {}
        
        print("\nInitializing clients / 初始化客户端...")
        for client_id in tqdm(range(self.num_clients), desc="Creating clients"):
            # 获取客户端数据 / Get client data
            train_loader = self.data_loader.get_client_train_dataloader(client_id)
            test_loader = self.data_loader.get_client_test_dataloader(client_id)
            
            # 获取样本数量 / Get sample counts
            num_train = self.data_loader.get_num_train_samples(client_id)
            num_test = self.data_loader.get_num_test_samples(client_id)
            
            # 创建客户端 / Create client
            client = FederatedClient(
                client_id=client_id,
                model=self.model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                num_train_samples=num_train,
                num_test_samples=num_test,
                device=self.device
            )
            
            self.clients[client_id] = client
        
        print(f"✓ Created {self.num_clients} clients")
    
    def _initialize_incentive_system(self):
        """初始化激励系统 / Initialize incentive system"""
        # 时间片管理器 / Time slice manager
        self.time_slice_manager = TimeSliceManager(
            slice_type=self.time_slice_type,
            rounds_per_slice=IncentiveConfig.ROUNDS_PER_SLICE,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        # 会员系统 / Membership system
        self.membership_system = MembershipSystem(
            level_thresholds=IncentiveConfig.LEVEL_THRESHOLDS,
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS
        )
        
        # 初始化客户端会员信息 / Initialize client membership
        for client_id in range(self.num_clients):
            self.membership_system.initialize_client(client_id)
    
    def _initialize_metrics(self):
        """初始化指标 / Initialize metrics"""
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
    
    def calculate_standalone_baselines(self, epochs: int = 10):
        """
        计算独立训练基准 / Calculate standalone baselines
        每个客户端在各自的测试集上评估
        Each client evaluated on their own test set
        
        Args:
            epochs: 独立训练轮次 / Standalone training epochs
        """
        print(f"\n{'='*70}")
        print(f"Computing Standalone Baselines / 计算独立训练基准")
        print(f"{'='*70}")
        
        for client_id, client in tqdm(self.clients.items(), 
                                     desc="Standalone training"):
            standalone_acc, standalone_loss = client.train_standalone(epochs=epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        print("✓ Standalone baselines computed")
    
    def run_single_round(self, round_num: int) -> Dict:
        """
        运行单轮训练 / Run single round of training
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            轮次指标 / Round metrics
        """
        round_start_time = time.time()
        
        # 所有客户端参与 / All clients participate
        selected_clients = list(range(self.num_clients))
        
        # 重置服务器 / Reset server
        self.server.reset_round()
        
        # 存储客户端准确率 / Store client accuracies
        client_accuracies = {}
        
        # 客户端训练 / Client training
        if round_num % 10 == 0 or round_num == 1:
            print(f"\nRound {round_num}: Training {len(selected_clients)} clients...")
        
        for client_id in tqdm(selected_clients, desc=f"Round {round_num}", 
                            disable=(round_num % 10 != 0 and round_num != 1)):
            client = self.clients[client_id]
            
            # 获取个性化模型 / Get personalized model
            if round_num == 1:
                model_weights = self.server.get_global_model_weights()
            else:
                if hasattr(self, 'personalized_models') and client_id in self.personalized_models:
                    model_weights = self.personalized_models[client_id]
                else:
                    model_weights = self.server.get_global_model_weights()
            
            # 训练 / Train
            updated_weights, train_info = client.train_federated(
                global_weights=model_weights,
                epochs=FederatedConfig.LOCAL_EPOCHS,
                lr=FederatedConfig.LEARNING_RATE
            )
            
            # 收集更新 / Collect updates
            self.server.collect_client_updates(client_id, updated_weights, train_info)
            
            # 记录准确率 / Record accuracy
            federated_acc = train_info['federated_accuracy']
            self.metrics_calculator.record_federated_accuracy(client_id, federated_acc)
            client_accuracies[client_id] = federated_acc
        
        # 分发个性化模型 / Distribute personalized models
        self.personalized_models = self.server.distribute_personalized_models(round_num)
        
        # 更新全局模型 / Update global model
        self.server.update_global_model()
        
        # 更新激励系统 / Update incentive system
        contributions = self.server.client_contributions
        for client_id, contribution in contributions.items():
            # 更新时间片积分 / Update time slice points
            current_slice = self.time_slice_manager.get_current_slice(round_num)
            self.time_slice_manager.update_client_slice_points(
                client_id, round_num, contribution * 1000
            )
            
            # 获取有效积分 / Get active points
            active_points = self.time_slice_manager.get_active_points(client_id, round_num)
            
            # 更新会员等级 / Update membership level
            new_level = self.membership_system.update_membership_level(client_id, active_points)
            self.clients[client_id].update_membership_level(new_level)
        
        # 计算轮次时间 / Calculate round time
        round_time = time.time() - round_start_time
        
        # 记录指标 / Record metrics
        round_metrics = {
            'round': round_num,
            'time_consumption': round_time,
            'num_selected_clients': len(selected_clients),
            'contributions': self.server.client_contributions.copy(),
            'client_accuracies': client_accuracies.copy()
        }
        
        self.metrics_calculator.record_round(round_metrics)
        
        return round_metrics
    
    def run_experiment(self, experiment_name: str) -> Dict:
        """
        运行完整实验 / Run complete experiment
        
        Args:
            experiment_name: 实验名称 / Experiment name
            
        Returns:
            实验结果 / Experiment results
        """
        print(f"\n{'='*70}")
        print(f"Starting Experiment: {experiment_name}")
        print(f"{'='*70}")
        
        # 计算独立训练基准 / Calculate standalone baselines
        self.calculate_standalone_baselines(epochs=10)
        
        # 联邦学习训练 / Federated learning training
        print(f"\n{'='*70}")
        print(f"Federated Learning Training / 联邦学习训练")
        print(f"{'='*70}")
        
        for round_num in range(1, self.num_rounds + 1):
            round_metrics = self.run_single_round(round_num)
            
            # 打印进度 / Print progress
            if round_num % 10 == 0 or round_num == 1:
                print(f"\nRound {round_num}/{self.num_rounds}")
                
                if 'client_accuracies' in round_metrics:
                    client_accs = list(round_metrics['client_accuracies'].values())
                    print(f"  Avg Accuracy: {np.mean(client_accs):.4f}")
                    print(f"  Max Accuracy: {np.max(client_accs):.4f}")
                    print(f"  Min Accuracy: {np.min(client_accs):.4f}")
                
                print(f"  Time: {round_metrics['time_consumption']:.2f}s")
                
                if 'contributions' in round_metrics and round_metrics['contributions']:
                    contribs = list(round_metrics['contributions'].values())
                    print(f"  Contribution - Mean: {np.mean(contribs):.4f}, "
                          f"Std: {np.std(contribs):.4f}")
        
        # 计算最终指标 / Calculate final metrics
        final_metrics = self.metrics_calculator.calculate_final_metrics()
        
        # 生成可视化 / Generate visualizations
        self._generate_visualizations(experiment_name)
        
        # 保存结果 / Save results
        self._save_results(experiment_name, final_metrics)
        
        # 打印摘要 / Print summary
        self.metrics_calculator.print_summary()
        
        return final_metrics
    
    def _generate_visualizations(self, experiment_name: str):
        """生成可视化 / Generate visualizations"""
        print("\nGenerating visualizations / 生成可视化...")
        
        # PCC散点图 / PCC scatter plot
        pcc_data = self.metrics_calculator.get_pcc_visualization_data()
        if pcc_data:
            pcc_value, pcc_details = self.metrics_calculator.calculate_pcc()
            self.visualizer.plot_pcc_scatter(
                pcc_data['standalone_accuracies'],
                pcc_data['federated_accuracies'],
                pcc_value,
                pcc_details.get('p_value', 0),
                experiment_name
            )
        
        # 训练曲线 / Training curves
        contributions_history = []
        for round_metric in self.metrics_calculator.round_metrics:
            if 'contributions' in round_metric and round_metric['contributions']:
                contributions_history.append(round_metric['contributions'])
            else:
                contributions_history.append({})
        
        metrics_history = {
            'rounds': list(range(1, len(self.metrics_calculator.avg_client_accuracies) + 1)),
            'avg_client_accuracy': self.metrics_calculator.avg_client_accuracies,
            'max_client_accuracy': self.metrics_calculator.max_client_accuracies,
            'time_per_round': self.metrics_calculator.time_consumptions,
            'contributions': contributions_history
        }
        
        self.visualizer.plot_training_curves(metrics_history, experiment_name)
        
        print("✓ Visualizations generated")
    
    def _save_results(self, experiment_name: str, final_metrics: Dict):
        """保存结果 / Save results"""
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存指标 / Save metrics
        metrics_path = os.path.join(results_dir, f"{experiment_name}_metrics.json")
        self.metrics_calculator.save_metrics(metrics_path)
        
        # 保存配置 / Save configuration
        config_path = os.path.join(results_dir, f"{experiment_name}_config.json")
        config_data = {
            'dataset': self.dataset_name,
            'num_clients': self.num_clients,
            'num_rounds': self.num_rounds,
            'distribution': self.distribution,
            'time_slice_type': self.time_slice_type,
            'data_info': {
                'original_train_samples': self.data_loader.num_train_samples,
                'original_test_samples': self.data_loader.num_test_samples,
                'total_train_allocated': sum(self.data_loader.get_num_train_samples(i) 
                                            for i in range(self.num_clients)),
                'total_test_allocated': sum(self.data_loader.get_num_test_samples(i) 
                                           for i in range(self.num_clients))
            },
            'final_metrics': final_metrics
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {results_dir}")


def compare_experiments(configurations: List[Dict]) -> Dict:
    """比较不同配置的实验 / Compare experiments"""
    comparison_results = {}
    
    for config in configurations:
        print(f"\n{'='*70}")
        print(f"Running: {config['name']}")
        print(f"{'='*70}")
        
        runner = ExperimentRunner(
            dataset_name=config['dataset'],
            num_clients=config['num_clients'],
            num_rounds=config['num_rounds'],
            distribution=config['distribution'],
            time_slice_type=config['time_slice_type'],
            device=config.get('device', DEVICE)
        )
        
        results = runner.run_experiment(config['name'])
        comparison_results[config['name']] = results
    
    # 打印比较 / Print comparison
    print(f"\n{'='*70}")
    print("Comparison Summary / 比较摘要")
    print(f"{'='*70}")
    
    for name, results in comparison_results.items():
        print(f"\n{name}:")
        print(f"  Avg Accuracy: {results['client_accuracy']['avg_final']:.4f}")
        print(f"  Max Accuracy: {results['client_accuracy']['max_final']:.4f}")
        print(f"  Total Time: {results['time_consumption']['total']:.2f}s")
        print(f"  PCC: {results['pcc']:.4f}")
    
    return comparison_results