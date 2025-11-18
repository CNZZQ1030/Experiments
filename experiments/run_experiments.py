"""
experiments/run_experiments.py (Updated for CGSV and Relative Ranking)
实验运行器 - CGSV贡献度 + 相对排名会员系统
Experiment Runner - CGSV contribution + Relative ranking membership system

核心改进 / Core Improvements:
1. CGSV代替AMAC / CGSV instead of AMAC
2. 相对排名会员系统 / Relative ranking membership system
3. 改进的贡献度-等级-模型映射 / Improved contribution-level-model mapping
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
    
    核心改进 / Core Improvements:
    1. 使用CGSV计算贡献度 / Use CGSV for contribution calculation
    2. 基于相对排名的会员系统 / Membership system based on relative ranking
    3. 每轮动态更新所有客户端等级 / Update all client levels dynamically each round
    """
    
    def __init__(self, dataset_name: str, num_clients: int, num_rounds: int,
                 distribution: str = "iid", time_slice_type: str = "rounds",
                 rounds_per_slice: int = None, device: torch.device = None,
                 use_relative_normalization: bool = True):
        """
        初始化实验运行器 / Initialize experiment runner
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            num_clients: 客户端数量 / Number of clients
            num_rounds: 训练轮次 / Training rounds
            distribution: 数据分布 (iid/non-iid) / Data distribution
            time_slice_type: 时间片类型 / Time slice type
            rounds_per_slice: 每个时间片的轮次数 / Rounds per slice
            device: 计算设备 / Computing device
            use_relative_normalization: 是否使用相对归一化 / Use relative normalization
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.distribution = distribution
        self.time_slice_type = time_slice_type
        self.rounds_per_slice = rounds_per_slice or IncentiveConfig.ROUNDS_PER_SLICE
        self.device = device or DEVICE
        self.use_relative_normalization = use_relative_normalization
        
        print(f"\n{'='*70}")
        print(f"Experiment Initialization / 实验初始化")
        print(f"{'='*70}")
        print(f"Dataset / 数据集: {dataset_name}")
        print(f"Clients / 客户端数: {num_clients}")
        print(f"Rounds / 轮次: {num_rounds}")
        print(f"Distribution / 分布: {distribution}")
        print(f"Time Slice / 时间片: {time_slice_type}")
        print(f"Rounds per Slice / 每时间片轮次: {self.rounds_per_slice}")
        print(f"Device / 设备: {self.device}")
        print(f"\nImproved Strategy / 改进策略:")
        print(f"  1. CGSV Contribution Calculation / CGSV贡献度计算")
        print(f"  2. Relative Ranking Membership / 相对排名会员制")
        print(f"  3. Contribution Normalization / 贡献度归一化: {use_relative_normalization}")
        
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
        self.server = FederatedServer(
            self.model, 
            self.device,
            use_relative_normalization=self.use_relative_normalization
        )
    
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
        
        # 会员系统（相对排名版本）/ Membership system (relative ranking version)
        self.membership_system = MembershipSystem(
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS
            # 注意：不再需要level_thresholds，使用相对排名
            # Note: No longer need level_thresholds, use relative ranking
        )
        
        # 初始化客户端会员信息 / Initialize client membership
        for client_id in range(self.num_clients):
            self.membership_system.initialize_client(client_id)
        
        print(f"\n✓ Incentive system initialized")
        print(f"  Strategy: CGSV + Relative Ranking + Periodic expiration")
        print(f"  Time slice type: {self.time_slice_type}")
        print(f"  Rounds per slice: {self.rounds_per_slice}")
        print(f"  Validity: {IncentiveConfig.POINTS_VALIDITY_SLICES} slices")
    
    def _initialize_metrics(self):
        """初始化指标 / Initialize metrics"""
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
    
    def calculate_standalone_baselines(self, epochs: int = None):
        """
        计算独立训练基准 / Calculate standalone baselines
        每个客户端在各自的测试集上评估
        Each client evaluated on their own test set
    
        Args:
            epochs: 独立训练轮次 / Standalone training epochs
        """
        if epochs is None:
            epochs = FederatedConfig.STANDALONE_EPOCHS

        print(f"\n{'='*70}")
        print(f"Computing Standalone Baselines / 计算独立训练基准")
        print(f"Each client trains independently for {epochs} epochs")
        print(f"{'='*70}")
        
        for client_id, client in tqdm(self.clients.items(), 
                                     desc="Standalone training"):
            standalone_acc, standalone_loss = client.train_standalone(epochs=epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        print("✓ Standalone baselines computed")

"""
experiments/run_experiments.py (Part 2: Training Logic)
实验运行器 - 第2部分：核心训练逻辑
Experiment Runner - Part 2: Core Training Logic
"""

    def run_single_round(self, round_num: int) -> Dict:
        """
        运行单轮训练 / Run single round of training
        
        核心改进 / Core Improvements:
        1. 使用CGSV计算贡献度 / Use CGSV for contribution
        2. 基于相对排名更新会员等级 / Update membership by relative ranking
        3. 改进的贡献度积分转换 / Improved contribution-to-points conversion
        
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
        
        # 显示详细信息的条件 / Show details condition
        show_details = (round_num % max(1, self.num_rounds // 10) == 0) or round_num == 1 or round_num == self.num_rounds
        
        if show_details:
            print(f"\n{'='*70}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*70}")
        
        # =====================================================================
        # 步骤1: 客户端训练 / Step 1: Client Training
        # =====================================================================
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num}/{self.num_rounds}",
                            leave=False):
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
        
        # =====================================================================
        # 步骤2: 计算贡献度（CGSV + 归一化）/ Step 2: Calculate Contributions
        # =====================================================================
        normalized_contributions, raw_contributions = self.server.calculate_all_contributions(round_num)
        
        # =====================================================================
        # 步骤3: 实时积分累加 / Step 3: Real-time Point Accumulation
        # =====================================================================
        # 将归一化贡献度转换为积分并累加到当前时间片
        # Convert normalized contribution to points and add to current slice
        all_active_points = {}
        for client_id, contribution in normalized_contributions.items():
            active_points = self.time_slice_manager.add_contribution_points(
                client_id, round_num, contribution
            )
            all_active_points[client_id] = active_points
        
        # =====================================================================
        # 步骤4: 基于相对排名更新所有客户端的会员等级 / Step 4: Update Membership by Ranking
        # =====================================================================
        # 这是核心改进！使用相对排名代替绝对阈值
        # This is the core improvement! Use relative ranking instead of absolute thresholds
        new_levels = self.membership_system.update_all_memberships_by_ranking(all_active_points)
        
        # 更新客户端的会员等级 / Update client membership levels
        for client_id, new_level in new_levels.items():
            self.clients[client_id].update_membership_level(new_level)
        
        # =====================================================================
        # 步骤5: 阶段性清理过期积分 / Step 5: Periodic Expiration Cleanup
        # =====================================================================
        current_slice = self.time_slice_manager.get_current_slice(round_num)
        if round_num > 1:
            prev_slice = self.time_slice_manager.get_current_slice(round_num - 1)
            if current_slice != prev_slice:
                cleaned = self.time_slice_manager.clean_expired_points(round_num)
                if cleaned and show_details:
                    print(f"Time slice changed: {prev_slice} → {current_slice}")
                    print(f"Cleaned expired points from {len(cleaned)} clients")
                    # 时间片切换后重新计算等级
                    updated_points = self.time_slice_manager.get_all_client_active_points(round_num)
                    new_levels = self.membership_system.update_all_memberships_by_ranking(updated_points)
                    for client_id, new_level in new_levels.items():
                        self.clients[client_id].update_membership_level(new_level)
        
        # =====================================================================
        # 步骤6: 分发个性化模型 / Step 6: Distribute Personalized Models
        # =====================================================================
        self.personalized_models = self.server.distribute_personalized_models(round_num)
        
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
            
            # 显示贡献度统计 / Show contribution statistics
            if normalized_contributions:
                norm_contribs = list(normalized_contributions.values())
                raw_contribs = list(raw_contributions.values())
                print(f"  Raw CGSV - Mean: {np.mean(raw_contribs):.4f}, Std: {np.std(raw_contribs):.4f}")
                if self.use_relative_normalization:
                    print(f"  Normalized - Mean: {np.mean(norm_contribs):.4f}, Std: {np.std(norm_contribs):.4f}")
            
            # 显示积分和等级统计 / Show points and level statistics
            if all_active_points:
                points_values = list(all_active_points.values())
                print(f"  Active Points - Mean: {np.mean(points_values):.2f}, Max: {np.max(points_values):.2f}")
            
            # 打印会员等级分布 / Print membership distribution
            if round_num % 10 == 0 or round_num == self.num_rounds:
                self.membership_system.print_membership_distribution()
        
        # 记录指标 / Record metrics
        round_metrics = {
            'round': round_num,
            'time_consumption': round_time,
            'num_selected_clients': len(selected_clients),
            'normalized_contributions': normalized_contributions.copy(),
            'raw_contributions': raw_contributions.copy(),
            'client_accuracies': client_accuracies.copy(),
            'current_slice': current_slice,
            'active_points': all_active_points.copy(),
            'membership_levels': {cid: self.clients[cid].membership_level for cid in self.clients}
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
        
        # 计算独立训练基准
        self.calculate_standalone_baselines()
        
        # 联邦学习训练 / Federated learning training
        print(f"\n{'='*70}")
        print(f"Federated Learning Training / 联邦学习训练")
        print(f"Total Rounds: {self.num_rounds}")
        print(f"Local Epochs per Round: {FederatedConfig.LOCAL_EPOCHS}")
        print(f"{'='*70}")
        
        for round_num in range(1, self.num_rounds + 1):
            round_metrics = self.run_single_round(round_num)
        
        print(f"\n{'='*70}")
        print(f"Training Complete / 训练完成")
        print(f"{'='*70}")
        
        # 打印时间片统计 / Print time slice statistics
        self.time_slice_manager.print_summary(self.num_rounds)
        
        # 打印贡献度统计 / Print contribution statistics
        self.server.print_contribution_summary()
        
        # 打印最终会员分布 / Print final membership distribution
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
    
"""
experiments/run_experiments.py (Part 3: Visualization and Saving)
实验运行器 - 第3部分：可视化和结果保存
Experiment Runner - Part 3: Visualization and Saving
"""

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
        # 准备贡献度历史（使用归一化后的）
        contributions_history = []
        raw_contributions_history = []
        for round_metric in self.metrics_calculator.round_metrics:
            if 'normalized_contributions' in round_metric:
                contributions_history.append(round_metric['normalized_contributions'])
            else:
                contributions_history.append({})
            
            if 'raw_contributions' in round_metric:
                raw_contributions_history.append(round_metric['raw_contributions'])
            else:
                raw_contributions_history.append({})
        
        metrics_history = {
            'rounds': list(range(1, len(self.metrics_calculator.avg_client_accuracies) + 1)),
            'avg_client_accuracy': self.metrics_calculator.avg_client_accuracies,
            'max_client_accuracy': self.metrics_calculator.max_client_accuracies,
            'time_per_round': self.metrics_calculator.time_consumptions,
            'contributions': contributions_history,  # 归一化贡献度
            'raw_contributions': raw_contributions_history  # 原始CGSV
        }
        
        self.visualizer.plot_training_curves(metrics_history, experiment_name)
        
        print("✓ Visualizations generated")
    
    def _save_results(self, experiment_name: str, final_metrics: Dict):
        """保存结果 / Save results"""
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存指标 / Save metrics
        metrics_path = os.path.join(results_dir, f"{experiment_name}_metrics.json")
        
        # 准备保存的数据
        save_data = {
            'final_metrics': final_metrics,
            'configuration': {
                'dataset': self.dataset_name,
                'num_clients': self.num_clients,
                'num_rounds': self.num_rounds,
                'distribution': self.distribution,
                'time_slice_type': self.time_slice_type,
                'rounds_per_slice': self.rounds_per_slice,
                'standalone_epochs': FederatedConfig.STANDALONE_EPOCHS,
                'local_epochs_per_round': FederatedConfig.LOCAL_EPOCHS,
                'use_relative_normalization': self.use_relative_normalization,
                'contribution_method': 'CGSV',
                'membership_method': 'Relative Ranking'
            },
            'data_info': {
                'original_train_samples': self.data_loader.num_train_samples,
                'original_test_samples': self.data_loader.num_test_samples,
                'total_train_allocated': sum(self.data_loader.get_num_train_samples(i) 
                                        for i in range(self.num_clients)),
                'total_test_allocated': sum(self.data_loader.get_num_test_samples(i) 
                                       for i in range(self.num_clients))
            },
            'round_metrics': self.metrics_calculator.round_metrics,
            'membership_statistics': self.membership_system.get_membership_statistics(),
            'contribution_statistics': self.server.get_contribution_statistics()
        }
        
        # 保存为JSON
        with open(metrics_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"✓ Results saved to: {metrics_path}")


def compare_experiments(configurations: List[Dict]) -> Dict:
    """
    比较不同配置的实验 / Compare experiments
    
    Args:
        configurations: 实验配置列表 / List of experiment configurations
        
    Returns:
        比较结果 / Comparison results
    """
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
            rounds_per_slice=config.get('rounds_per_slice'),
            device=config.get('device', DEVICE),
            use_relative_normalization=config.get('use_relative_normalization', True)
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
        
        # 打印会员等级分布
        if 'membership_statistics' in results:
            print(f"  Membership Distribution:")
            level_dist = results['membership_statistics']['level_percentages']
            for level in ['diamond', 'gold', 'silver', 'bronze']:
                if level in level_dist:
                    print(f"    {level.capitalize()}: {level_dist[level]:.1f}%")
    
    return comparison_results