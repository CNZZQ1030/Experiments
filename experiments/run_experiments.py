"""
experiments/run_experiments.py
实验运行模块 - 支持基于CGSV的激励机制
Experiment Running Module - Supporting CGSV-based Incentive Mechanism  
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import copy
from tqdm import tqdm

# 导入项目模块 / Import project modules
from datasets.data_loader import FederatedDataLoader
from models.cnn_model import ModelFactory
from federated.client import FederatedClient
from federated.server import FederatedServer
from incentive.membership import MembershipSystem
from incentive.points_calculator import PointsCalculator, CGSVCalculator
from incentive.time_slice import TimeSliceManager
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer
from config import IncentiveConfig, DatasetConfig


class ExperimentRunner:
    """
    实验运行器 / Experiment Runner
    负责运行和管理基于CGSV的联邦学习实验
    Responsible for running and managing CGSV-based federated learning experiments
    """
    
    def __init__(self, dataset_name: str, num_clients: int, num_rounds: int,
                 clients_per_round: int, time_slice_type: str,
                 distribution: str = "iid", device: torch.device = torch.device("cpu")):
        """
        初始化实验运行器 / Initialize experiment runner
        
        Args:
            dataset_name: 数据集名称 / Dataset name
            num_clients: 客户端数量 / Number of clients
            num_rounds: 训练轮次 / Number of training rounds
            clients_per_round: 每轮选择的客户端数 / Clients per round
            time_slice_type: 时间片类型 / Time slice type
            distribution: 数据分布 / Data distribution
            device: 计算设备 / Computing device
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.time_slice_type = time_slice_type
        self.distribution = distribution
        self.device = device
        
        # 初始化组件 / Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """初始化实验组件 / Initialize experiment components"""
        # 加载数据 / Load data
        self.data_loader = FederatedDataLoader(
            dataset_name=self.dataset_name,
            num_clients=self.num_clients,
            batch_size=32,
            distribution=self.distribution,
            alpha=0.5
        )
        
        # 创建模型 / Create model
        num_classes = DatasetConfig.NUM_CLASSES[self.dataset_name]
        self.model = ModelFactory.create_model(self.dataset_name, num_classes)
        
        # 创建服务器 / Create server
        self.server = FederatedServer(self.model, self.device)
        
        # 创建客户端 / Create clients
        self.clients = {}
        for client_id in range(self.num_clients):
            client_dataloader = self.data_loader.get_client_dataloader(client_id)
            self.clients[client_id] = FederatedClient(
                client_id=client_id,
                model=copy.deepcopy(self.model),
                dataloader=client_dataloader,
                device=self.device
            )
        
        # 初始化激励系统 / Initialize incentive system
        self.membership_system = MembershipSystem(
            level_thresholds=IncentiveConfig.LEVEL_THRESHOLDS,
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS
        )
        
        # 使用新的CGSV积分计算器 / Use new CGSV points calculator
        self.points_calculator = PointsCalculator()
        self.cgsv_calculator = CGSVCalculator()
        
        self.time_slice_manager = TimeSliceManager(
            slice_type=self.time_slice_type,
            rounds_per_slice=IncentiveConfig.ROUNDS_PER_SLICE,
            days_per_slice=IncentiveConfig.DAYS_PER_SLICE,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        # 初始化指标计算器和可视化器 / Initialize metrics calculator and visualizer
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # 存储客户端贡献度历史 / Store client contribution history
        self.client_contributions_history = []
        self.client_performances_history = []
        
    def select_clients(self, round_num: int, 
                      client_contributions: Dict[int, float] = None) -> List[int]:
        """
        选择客户端参与训练 / Select clients for training
        基于贡献度和会员等级进行选择 / Select based on contribution and membership level
        
        Args:
            round_num: 当前轮次 / Current round
            client_contributions: 客户端贡献度 / Client contributions
            
        Returns:
            选中的客户端ID列表 / List of selected client IDs
        """
        # 获取客户端的优先级 / Get client priorities
        client_priorities = []
        
        for client_id in range(self.num_clients):
            membership_info = self.membership_system.get_client_membership_info(client_id)
            level = membership_info['level']
            points = membership_info['total_points']
            
            # 如果有贡献度信息，使用贡献度 / Use contribution if available
            if client_contributions and client_id in client_contributions:
                contribution = client_contributions[client_id]
            else:
                contribution = 0.1  # 默认贡献度 / Default contribution
            
            # 计算综合优先级 / Calculate composite priority
            level_multiplier = self.membership_system.level_multipliers[level]
            priority = (points * 0.3 + contribution * 10000 * 0.7) * level_multiplier
            
            client_priorities.append((client_id, priority))
        
        # 按优先级排序 / Sort by priority
        client_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # 选择客户端 / Select clients
        selected_clients = []
        
        # 优先选择高贡献度客户端 / Prioritize high contribution clients
        high_priority_count = min(self.clients_per_round // 2, len(client_priorities) // 5)
        for i in range(high_priority_count):
            selected_clients.append(client_priorities[i][0])
        
        # 剩余随机选择 / Random selection for remaining
        remaining_clients = [c[0] for c in client_priorities if c[0] not in selected_clients]
        remaining_slots = self.clients_per_round - len(selected_clients)
        
        if remaining_slots > 0 and remaining_clients:
            random_selected = np.random.choice(
                remaining_clients, 
                min(remaining_slots, len(remaining_clients)),
                replace=False
            )
            selected_clients.extend(random_selected.tolist())
        
        return selected_clients
    
    def run_single_round(self, round_num: int) -> Tuple[float, float]:
        """
        运行单轮训练 / Run single training round
        
        Args:
            round_num: 轮次号 / Round number
            
        Returns:
            准确率和损失 / Accuracy and loss
        """
        print(f"\n--- Round {round_num}/{self.num_rounds} ---")
        
        # 获取之前的贡献度信息用于客户端选择 / Get previous contributions for client selection
        prev_contributions = None
        if self.client_contributions_history:
            prev_contributions = self.client_contributions_history[-1]
        
        # 选择客户端 / Select clients
        selected_clients = self.select_clients(round_num, prev_contributions)
        print(f"Selected clients: {selected_clients}")
        
        # 存储上一轮的全局权重 / Store previous global weights
        prev_global_weights = copy.deepcopy(self.server.get_global_weights())
        
        # 客户端训练 / Client training
        client_weights = {}
        client_infos = {}
        client_gradients = {}
        
        for client_id in tqdm(selected_clients, desc="Training clients"):
            client = self.clients[client_id]
            
            # 获取该客户端的个性化模型 / Get personalized model for this client
            personalized_weights = self.server.get_client_model_weights(client_id)
            
            # 本地训练 / Local training
            updated_weights, train_info = client.train(
                global_weights=personalized_weights,
                epochs=5,
                lr=0.01
            )
            
            # 计算梯度向量 / Calculate gradient vector
            gradient_vector = self.cgsv_calculator.compute_gradient_vector(
                updated_weights, personalized_weights
            )
            
            # 记录信息 / Record information
            client_weights[client_id] = updated_weights
            client_infos[client_id] = train_info
            client_gradients[client_id] = gradient_vector
        
        # 计算全局梯度（所有客户端梯度的平均）/ Calculate global gradient (average of all client gradients)
        global_gradient = np.mean(list(client_gradients.values()), axis=0)
        
        # 使用CGSV计算客户端贡献度 / Calculate client contributions using CGSV
        client_contributions = self.cgsv_calculator.calculate_cgsv_contributions(
            client_gradients, global_gradient
        )
        
        # 计算积分 / Calculate points
        client_points = self.points_calculator.calculate_points_with_cgsv(
            client_gradients, global_gradient, client_infos
        )
        
        # 更新会员等级 / Update membership levels
        for client_id in selected_clients:
            # 更新时间片积分 / Update time slice points
            points = client_points[client_id]
            self.time_slice_manager.update_client_slice_points(client_id, round_num, points)
            
            # 获取有效积分 / Get active points
            active_points = self.time_slice_manager.get_active_points(client_id, round_num)
            
            # 更新会员等级 / Update membership level
            new_level = self.membership_system.update_membership_level(client_id, active_points)
            
            # 添加等级信息到client_infos / Add level info to client_infos
            client_infos[client_id]['membership_level'] = new_level
            client_infos[client_id]['contribution'] = client_contributions[client_id]
            client_infos[client_id]['level_multiplier'] = self.membership_system.level_multipliers[new_level]
            
            # 更新客户端参与信息 / Update client participation info
            client.update_participation(round_num, active_points, new_level)
        
        # 服务器聚合并生成差异化模型 / Server aggregation and differentiated model generation
        self.server.aggregate_models(client_weights, client_infos, client_contributions)
        
        # 评估全局模型 / Evaluate global model
        test_loader = self.data_loader.get_test_dataloader()
        accuracy, loss = self.server.evaluate_global_model(test_loader)
        
        print(f"Round {round_num} - Global Model - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # 评估客户端个性化模型 / Evaluate client personalized models
        client_performances = self.server.evaluate_client_models(test_loader, selected_clients)
        
        # 打印贡献度和性能信息 / Print contribution and performance info
        print(f"\nClient Contributions and Performances:")
        for client_id in selected_clients[:5]:  # 只显示前5个 / Show only first 5
            contribution = client_contributions.get(client_id, 0)
            if client_id in client_performances:
                perf_acc, perf_loss = client_performances[client_id]
                print(f"  Client {client_id}: Contribution={contribution:.4f}, "
                      f"Accuracy={perf_acc:.4f}, Loss={perf_loss:.4f}")
        
        # 计算质量差距 / Calculate quality gap
        if client_performances:
            perfs = [acc for acc, _ in client_performances.values()]
            quality_gap = max(perfs) - min(perfs) if perfs else 0
            self.metrics_calculator.record_quality_gap(quality_gap)
            print(f"Quality Gap: {quality_gap:.4f}")
        
        # 转换client_performances为只包含准确率的字典 / Convert to accuracy-only dict
        performance_dict = {cid: acc for cid, (acc, _) in client_performances.items()}
        
        # 更新激励机制评价指标 / Update incentive mechanism metrics
        self.metrics_calculator.update_incentive_metrics(
            round_num=round_num,
            contributions=client_contributions,
            performances=performance_dict,
            rewards=client_points
        )
        
        # 记录贡献度和性能历史 / Record contribution and performance history
        self.client_contributions_history.append(client_contributions)
        self.client_performances_history.append(performance_dict)
        
        # 计算系统指标 / Calculate system metrics
        participation_rate = len(selected_clients) / self.num_clients
        
        # 计算客户端参与度 / Calculate client participation
        client_participations = {}
        for client_id in selected_clients:
            participation = self.metrics_calculator.calculate_client_participation(
                client_id=client_id,
                rounds_participated=len(self.clients[client_id].participation_rounds),
                total_rounds=round_num,
                contribution=client_contributions.get(client_id, 0)
            )
            client_participations[client_id] = participation
        
        # 计算系统活跃度 / Calculate system activity
        system_activity = self.metrics_calculator.calculate_system_activity(
            client_participations=client_participations,
            active_clients=len(selected_clients),
            total_clients=self.num_clients
        )
        
        # 获取会员统计 / Get membership statistics
        membership_stats = self.membership_system.get_membership_statistics()
        
        # 更新指标历史 / Update metrics history
        self.metrics_calculator.update_metrics(
            round_num=round_num,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            system_activity=system_activity,
            level_distribution=membership_stats['level_distribution'],
            points_stats={'avg_points': membership_stats['average_points']}
        )
        
        # 清理过期积分 / Clean expired points
        if round_num % 10 == 0:
            self.time_slice_manager.clean_expired_points(round_num)
        
        return accuracy, loss
    
    def run_single_experiment(self, experiment_name: str, 
                             num_runs: int = 1) -> Dict:
        """
        运行单个实验 / Run single experiment
        
        Args:
            experiment_name: 实验名称 / Experiment name
            num_runs: 运行次数 / Number of runs
            
        Returns:
            实验结果 / Experiment results
        """
        all_results = []
        
        for run in range(num_runs):
            print(f"\n{'='*50}")
            print(f"Run {run + 1}/{num_runs}")
            print('='*50)
            
            # 重新初始化组件 / Reinitialize components
            if run > 0:
                self._initialize_components()
            
            # 运行所有轮次 / Run all rounds
            for round_num in range(1, self.num_rounds + 1):
                accuracy, loss = self.run_single_round(round_num)
                
                # 定期保存检查点 / Periodically save checkpoints
                if round_num % 10 == 0:
                    checkpoint_path = f"outputs/checkpoints/{experiment_name}_round_{round_num}.pt"
                    self.server.save_checkpoint(round_num, checkpoint_path)
            
            # 收集结果 / Collect results
            metrics_summary = self.metrics_calculator.get_metrics_summary()
            metrics_summary['convergence_round'] = self.metrics_calculator.calculate_convergence_round()
            all_results.append(metrics_summary)
            
            # 生成可视化 / Generate visualizations
            # 训练曲线 / Training curves
            self.visualizer.plot_training_curves(
                self.metrics_calculator.metrics_history,
                f"{experiment_name}_run_{run + 1}"
            )
            
            # 激励机制评价指标 / Incentive mechanism metrics
            self.visualizer.plot_incentive_metrics(
                self.metrics_calculator.metrics_history,
                f"{experiment_name}_run_{run + 1}"
            )
            
            # 贡献度-性能散点图 / Contribution-performance scatter plot
            self.visualizer.plot_contribution_performance_scatter(
                self.client_contributions_history,
                self.client_performances_history,
                f"{experiment_name}_run_{run + 1}"
            )
            
            # 综合报告 / Comprehensive report
            self.visualizer.create_comprehensive_report(
                metrics_summary,
                f"{experiment_name}_run_{run + 1}"
            )
        
        # 汇总多次运行的结果 / Aggregate results from multiple runs
        final_results = self._aggregate_run_results(all_results)
        final_results['experiment_name'] = experiment_name
        final_results['num_runs'] = num_runs
        
        return final_results
    
    def _aggregate_run_results(self, all_results: List[Dict]) -> Dict:
        """
        汇总多次运行的结果 / Aggregate results from multiple runs
        
        Args:
            all_results: 所有运行的结果 / Results from all runs
            
        Returns:
            汇总结果 / Aggregated results
        """
        aggregated = {}
        
        # 计算所有指标的平均值和标准差 / Calculate mean and std for all metrics
        metrics = ['accuracy_final', 'accuracy_avg', 'loss_final', 'loss_avg',
                  'participation_rate_final', 'participation_rate_avg',
                  'system_activity_final', 'system_activity_avg', 'convergence_round',
                  'contribution_reward_correlation_final', 'contribution_reward_correlation_avg',
                  'fairness_index_final', 'fairness_index_avg',
                  'incentive_effectiveness_final', 'incentive_effectiveness_avg',
                  'quality_gap_final', 'quality_gap_avg']
        
        for metric in metrics:
            values = [r.get(metric, 0) for r in all_results if metric in r]
            if values:
                aggregated[f'{metric}'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated