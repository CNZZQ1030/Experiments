"""
experiments/run_experiments.py (Updated for UPSM Implementation)
实验运行器 - 实现PDF文档中的完整工作流程
Experiment Runner - Implements complete workflow from PDF

核心流程 / Core Workflow (PDF Algorithm 1):
1. 计算瞬时贡献 CGSV / Calculate instantaneous contribution
2. 时间片滑动窗口累计 / Time-slice sliding window accumulation
3. 基于相对排名的会员定级 / Relative ranking based membership
4. UPSM差异化模型分发 / UPSM differentiated model distribution
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
    
    实现PDF文档Algorithm 1的完整流程 / Implements complete PDF Algorithm 1
    """
    
    def __init__(self, dataset_name: str, num_clients: int, num_rounds: int,
                 distribution: str = "iid", time_slice_type: str = "rounds",
                 rounds_per_slice: int = None, device: torch.device = None,
                 alpha: float = 0.5):
        """
        初始化实验运行器 / Initialize experiment runner
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            num_clients: 客户端数量 / Number of clients
            num_rounds: 训练轮次 / Training rounds
            distribution: 数据分布 / Data distribution
            time_slice_type: 时间片类型 / Time slice type
            rounds_per_slice: 每个时间片的轮次数 (τ) / Rounds per slice
            device: 计算设备 / Computing device
            alpha: Dirichlet分布参数 / Dirichlet parameter
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.distribution = distribution
        self.time_slice_type = time_slice_type
        self.rounds_per_slice = rounds_per_slice or IncentiveConfig.ROUNDS_PER_SLICE
        self.device = device or DEVICE
        self.alpha = alpha
        
        print(f"\n{'='*70}")
        print(f"Experiment Initialization / 实验初始化")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Clients: {num_clients}")
        print(f"Rounds: {num_rounds}")
        print(f"Distribution: {distribution}")
        print(f"Time Slice Type: {time_slice_type}")
        print(f"Rounds per Slice (τ): {self.rounds_per_slice}")
        print(f"Validity Window (W): {IncentiveConfig.POINTS_VALIDITY_SLICES}")
        print(f"Device: {self.device}")
        print(f"\nUPSM Configuration:")
        print(f"  Access Ratios (ρ): {IncentiveConfig.LEVEL_ACCESS_RATIOS}")
        print(f"  Selection Bias (β): {IncentiveConfig.LEVEL_SELECTION_BIAS}")
        
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
            alpha=self.alpha
        )
    
    def _initialize_model(self):
        """初始化模型 / Initialize model"""
        num_classes = DatasetConfig.NUM_CLASSES[self.dataset_name]
        input_channels = DatasetConfig.INPUT_SHAPE[self.dataset_name][0]
        
        if self.dataset_name == "sst":
            self.model = ModelFactory.create_model(
                self.dataset_name,
                num_classes=num_classes,
                vocab_size=DatasetConfig.SST_VOCAB_SIZE,
                embedding_dim=DatasetConfig.SST_EMBEDDING_DIM,
                max_seq_length=DatasetConfig.SST_MAX_SEQ_LENGTH
            )
        else:
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
            train_loader = self.data_loader.get_client_train_dataloader(client_id)
            test_loader = self.data_loader.get_client_test_dataloader(client_id)
            
            num_train = self.data_loader.get_num_train_samples(client_id)
            num_test = self.data_loader.get_num_test_samples(client_id)
            
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
            rounds_per_slice=self.rounds_per_slice,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        # 会员系统 / Membership system
        self.membership_system = MembershipSystem(
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS,
            ranking_percentiles=IncentiveConfig.LEVEL_PERCENTILES
        )
        
        # 初始化客户端会员 / Initialize client membership
        for client_id in range(self.num_clients):
            self.membership_system.initialize_client(client_id)
        
        print(f"\n✓ Incentive system initialized")
        print(f"  Strategy: CGSV + Time-Slice Window + Relative Ranking + UPSM")
    
    def _initialize_metrics(self):
        """初始化指标 / Initialize metrics"""
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
    
    def calculate_standalone_baselines(self, epochs: int = None):
        """
        计算独立训练基准 / Calculate standalone baselines
        
        Args:
            epochs: 独立训练轮次 / Standalone training epochs
        """
        if epochs is None:
            epochs = FederatedConfig.STANDALONE_EPOCHS

        print(f"\n{'='*70}")
        print(f"Computing Standalone Baselines")
        print(f"Each client trains independently for {epochs} epochs")
        print(f"{'='*70}")
        
        for client_id, client in tqdm(self.clients.items(), 
                                     desc="Standalone training"):
            standalone_acc, _ = client.train_standalone(epochs=epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        print("✓ Standalone baselines computed")

    def run_single_round(self, round_num: int) -> Dict:
        """
        运行单轮训练 / Run single round of training
        
        实现PDF Algorithm 1的完整流程 / Implements complete PDF Algorithm 1
        
        Args:
            round_num: 当前轮次 t / Current round
            
        Returns:
            轮次指标 / Round metrics
        """
        round_start_time = time.time()
        
        # 所有客户端参与 / All clients participate
        selected_clients = list(range(self.num_clients))
        
        # 重置服务器 / Reset server
        self.server.reset_round()
        
        # 存储准确率 / Store accuracies
        client_accuracies = {}
        
        # 显示详细信息条件 / Show details condition
        show_details = (round_num % max(1, self.num_rounds // 10) == 0) or \
                       round_num == 1 or round_num == self.num_rounds
        
        if show_details:
            print(f"\n{'='*70}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*70}")
        
        # =====================================================================
        # 步骤1: 客户端本地训练 / Step 1: Client Local Training
        # =====================================================================
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num}/{self.num_rounds}",
                            leave=False):
            client = self.clients[client_id]
            
            # 获取模型（第一轮用全局模型，之后用个性化模型）
            if round_num == 1:
                model_weights = self.server.get_global_model_weights()
            else:
                if hasattr(self, 'personalized_models') and client_id in self.personalized_models:
                    model_weights = self.personalized_models[client_id]
                else:
                    model_weights = self.server.get_global_model_weights()
            
            # 本地训练 / Local training
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
        
        # =====================================================================
        # 步骤2: 计算瞬时贡献 s_{i,t} / Step 2: Calculate Instantaneous Contribution
        # =====================================================================
        contributions = self.server.calculate_all_contributions(round_num)
        
        # =====================================================================
        # 步骤3: 时间片滑动窗口累计 R_{i,t} / Step 3: Time-Slice Window Accumulation
        # =====================================================================
        all_active_points = {}
        for client_id, contribution in contributions.items():
            # 将贡献度累加到当前时间片 / Add contribution to current time slice
            active_points = self.time_slice_manager.add_contribution_points(
                client_id, round_num, contribution
            )
            all_active_points[client_id] = active_points
        
        # =====================================================================
        # 步骤4: 基于相对排名更新会员等级 / Step 4: Update Membership by Ranking
        # =====================================================================
        new_levels = self.membership_system.update_all_memberships_by_ranking(all_active_points)
        
        # 更新客户端等级 / Update client levels
        for client_id, new_level in new_levels.items():
            self.clients[client_id].update_membership_level(new_level)
        
        # =====================================================================
        # 步骤5: 阶段性清理过期积分 / Step 5: Periodic Expiration Cleanup
        # =====================================================================
        current_slice = self.time_slice_manager.get_current_slice(round_num)
        if round_num > 1:
            prev_slice = self.time_slice_manager.get_current_slice(round_num - 1)
            if current_slice != prev_slice:
                # 时间片切换，清理过期积分 / Time slice changed, clean expired points
                cleaned = self.time_slice_manager.clean_expired_points(round_num)
                if cleaned and show_details:
                    print(f"Time slice changed: {prev_slice} → {current_slice}")
                    print(f"Cleaned expired points from {len(cleaned)} clients")
                
                # 重新计算等级 / Recalculate levels
                updated_points = self.time_slice_manager.get_all_client_active_points(round_num)
                new_levels = self.membership_system.update_all_memberships_by_ranking(updated_points)
                for client_id, new_level in new_levels.items():
                    self.clients[client_id].update_membership_level(new_level)
        
        # =====================================================================
        # 步骤6: UPSM差异化模型分发 / Step 6: UPSM Differentiated Model Distribution
        # =====================================================================
        self.personalized_models = self.server.distribute_personalized_models(new_levels)
        
        # =====================================================================
        # 步骤7: 更新全局模型 / Step 7: Update Global Model
        # =====================================================================
        self.server.update_global_model()
        
        # 计算轮次时间 / Calculate round time
        round_time = time.time() - round_start_time
        
        # 打印详细进度 / Print detailed progress
        if show_details:
            if client_accuracies:
                client_accs = list(client_accuracies.values())
                print(f"Results:")
                print(f"  Avg Accuracy: {np.mean(client_accs):.4f}")
                print(f"  Max Accuracy: {np.max(client_accs):.4f}")
                print(f"  Min Accuracy: {np.min(client_accs):.4f}")
            
            print(f"  Time: {round_time:.2f}s")
            
            # 贡献度统计 / Contribution statistics
            if contributions:
                contrib_values = list(contributions.values())
                print(f"  CGSV - Mean: {np.mean(contrib_values):.4f}, "
                      f"Std: {np.std(contrib_values):.4f}")
            
            # 积分统计 / Points statistics
            if all_active_points:
                points_values = list(all_active_points.values())
                print(f"  Active Points - Mean: {np.mean(points_values):.2f}, "
                      f"Max: {np.max(points_values):.2f}")
            
            # 会员分布 / Membership distribution
            if round_num % 10 == 0 or round_num == self.num_rounds:
                self.membership_system.print_membership_distribution()
        
        # 记录指标 / Record metrics
        round_metrics = {
            'round': round_num,
            'time_consumption': round_time,
            'num_selected_clients': len(selected_clients),
            'contributions': contributions.copy(),
            'client_accuracies': client_accuracies.copy(),
            'current_slice': current_slice,
            'active_points': all_active_points.copy(),
            'membership_levels': new_levels.copy()
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
        
        # 计算独立训练基准 / Compute standalone baselines
        self.calculate_standalone_baselines()
        
        # 联邦学习训练 / Federated learning training
        print(f"\n{'='*70}")
        print(f"Federated Learning Training")
        print(f"Total Rounds: {self.num_rounds}")
        print(f"{'='*70}")
        
        for round_num in range(1, self.num_rounds + 1):
            self.run_single_round(round_num)
        
        print(f"\n{'='*70}")
        print(f"Training Complete")
        print(f"{'='*70}")
        
        # 打印摘要 / Print summaries
        self.time_slice_manager.print_summary(self.num_rounds)
        self.server.print_contribution_summary()
        self.membership_system.print_membership_distribution()
        
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
        print("\nGenerating visualizations...")
        
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
            contributions_history.append(round_metric.get('contributions', {}))
        
        metrics_history = {
            'rounds': list(range(1, len(self.metrics_calculator.avg_client_accuracies) + 1)),
            'avg_client_accuracy': self.metrics_calculator.avg_client_accuracies,
            'max_client_accuracy': self.metrics_calculator.max_client_accuracies,
            'time_per_round': self.metrics_calculator.time_consumptions,
            'contributions': contributions_history,
            'raw_contributions': contributions_history  # 同一个数据
        }
        
        self.visualizer.plot_training_curves(metrics_history, experiment_name)
        
        print("✓ Visualizations generated")
    
    def _save_results(self, experiment_name: str, final_metrics: Dict):
        """保存结果 / Save results"""
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        metrics_path = os.path.join(results_dir, f"{experiment_name}_metrics.json")
        
        save_data = {
            'final_metrics': final_metrics,
            'configuration': {
                'dataset': self.dataset_name,
                'num_clients': self.num_clients,
                'num_rounds': self.num_rounds,
                'distribution': self.distribution,
                'time_slice_type': self.time_slice_type,
                'rounds_per_slice': self.rounds_per_slice,
                'validity_slices': IncentiveConfig.POINTS_VALIDITY_SLICES,
                'standalone_epochs': FederatedConfig.STANDALONE_EPOCHS,
                'local_epochs': FederatedConfig.LOCAL_EPOCHS,
                'contribution_method': 'CGSV',
                'membership_method': 'Relative Ranking',
                'distribution_method': 'UPSM'
            },
            'upsm_config': {
                'access_ratios': IncentiveConfig.LEVEL_ACCESS_RATIOS,
                'selection_bias': IncentiveConfig.LEVEL_SELECTION_BIAS
            },
            'round_metrics': self.metrics_calculator.round_metrics,
            'membership_statistics': self.membership_system.get_membership_statistics(),
            'contribution_statistics': self.server.get_contribution_statistics()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"✓ Results saved to: {metrics_path}")