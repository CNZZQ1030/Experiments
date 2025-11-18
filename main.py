"""
main.py (Complete Integration)
主程序 - 完整集成版本
Main Program - Complete Integration

整合所有改进：
Integrating all improvements:
1. CGSV贡献度计算（相对归一化）/ CGSV contribution calculation (relative normalization)
2. 相对排名会员系统 / Relative ranking membership system
3. IPR指标和可视化 / IPR metrics and visualization
4. 改进的差异化模型分发 / Improved differentiated model distribution
"""

import torch
import numpy as np
import random
import argparse
import os
import sys
from typing import Dict, List

# 添加项目路径 / Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FederatedConfig, IncentiveConfig, DatasetConfig, DEVICE
from datasets.data_loader import FederatedDataLoader
from models.cnn_model import ModelFactory
from federated.server import FederatedServer
from federated.client import FederatedClient
from incentive.time_slice import TimeSliceManager
from incentive.membership import MembershipSystem
from incentive.points_calculator import CGSVContributionCalculator
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer
import time
from tqdm import tqdm


def set_seed(seed: int = 42):
    """设置随机种子 / Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


class FederatedLearningExperiment:
    """
    联邦学习实验类 / Federated Learning Experiment Class
    完整集成CGSV + 相对排名 + IPR
    Complete integration of CGSV + Relative Ranking + IPR
    """
    
    def __init__(self, args):
        """
        初始化实验 / Initialize experiment
        
        Args:
            args: 命令行参数 / Command line arguments
        """
        self.args = args
        
        # 统一数据集名称为小写（兼容性处理）
        # Normalize dataset name to lowercase (compatibility)
        self.args.dataset = self.args.dataset.lower()
        
        # 设置随机种子 / Set random seed
        set_seed(args.seed)
        
        # 设备 / Device
        self.device = DEVICE
        
        # 实验名称 / Experiment name
        self.experiment_name = self._generate_experiment_name()
        
        print(f"\n{'='*80}")
        print(f"Federated Learning Experiment with CGSV + Relative Ranking + IPR")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Clients: {args.num_clients}")
        print(f"Rounds: {args.num_rounds}")
        print(f"Distribution: {args.distribution}")
        print(f"Device: {self.device}")
        print(f"{'='*80}")
        
        # 初始化组件 / Initialize components
        self._initialize_components()
        
    def _generate_experiment_name(self) -> str:
        """生成实验名称 / Generate experiment name"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{self.args.dataset}_{self.args.distribution}_" \
               f"c{self.args.num_clients}_r{self.args.num_rounds}_" \
               f"CGSV_RelRank_IPR_{timestamp}"
        return name
    
    def _initialize_components(self):
        """初始化所有组件 / Initialize all components"""
        print("\nInitializing components...")
        
        # 1. 数据加载器 / Data loader
        print("  [1/6] Loading data...")
        self.data_loader = FederatedDataLoader(
            dataset_name=self.args.dataset,
            num_clients=self.args.num_clients,
            batch_size=self.args.batch_size,
            distribution=self.args.distribution,
            alpha=self.args.alpha
        )
        
        # 2. 模型 / Model
        print("  [2/6] Creating model...")
        num_classes = DatasetConfig.NUM_CLASSES[self.args.dataset]
        input_channels = DatasetConfig.INPUT_SHAPE[self.args.dataset][0]
        self.model = ModelFactory.create_model(
            self.args.dataset,
            num_classes=num_classes,
            input_channels=input_channels
        )
        
        # 3. 服务器（集成CGSV和相对归一化）/ Server (with CGSV and relative normalization)
        print("  [3/6] Initializing server with CGSV...")
        self.server = FederatedServer(
            self.model,
            self.device,
            use_relative_normalization=self.args.use_relative_normalization
        )
        
        # 4. 客户端 / Clients
        print("  [4/6] Creating clients...")
        self.clients = {}
        for client_id in tqdm(range(self.args.num_clients), desc="    Creating clients", leave=False):
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
        
        # 5. 激励系统（相对排名会员系统）/ Incentive system (relative ranking membership)
        print("  [5/6] Initializing incentive system...")
        self.time_slice_manager = TimeSliceManager(
            slice_type=self.args.time_slice_type,
            rounds_per_slice=self.args.rounds_per_slice,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        self.membership_system = MembershipSystem(
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS
            # 不再需要level_thresholds，使用相对排名
            # No longer need level_thresholds, use relative ranking
        )
        
        for client_id in range(self.args.num_clients):
            self.membership_system.initialize_client(client_id)
        
        # 6. 评估和可视化（包含IPR）/ Evaluation and visualization (with IPR)
        print("  [6/6] Initializing metrics and visualization...")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(output_dir="outputs/figures")
        
        print("✓ All components initialized")
    
    def compute_standalone_baselines(self):
        """
        计算独立训练基准 / Compute standalone baselines
        每个客户端在自己的测试集上评估
        Each client evaluated on their own test set
        """
        print(f"\n{'='*80}")
        print("Computing Standalone Baselines")
        print(f"{'='*80}")
        print(f"Each client trains independently for {self.args.standalone_epochs} epochs")
        
        for client_id, client in tqdm(self.clients.items(), 
                                     desc="Standalone training"):
            standalone_acc, _ = client.train_standalone(epochs=self.args.standalone_epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        print("✓ Standalone baselines computed")
    
    def run_single_round(self, round_num: int) -> Dict:
        """
        运行单轮训练 / Run single round
        
        核心流程 / Core workflow:
        1. 客户端训练 / Client training
        2. CGSV贡献度计算（批量+归一化）/ CGSV calculation (batch + normalization)
        3. 积分累加 / Points accumulation
        4. 相对排名更新会员等级 / Update membership by relative ranking
        5. 分发个性化模型 / Distribute personalized models
        6. 更新全局模型 / Update global model
        7. IPR计算 / IPR calculation
        
        Args:
            round_num: 当前轮次 / Current round
            
        Returns:
            轮次指标 / Round metrics
        """
        round_start = time.time()
        
        # 选择客户端（这里全部参与）/ Select clients (all participate here)
        selected_clients = list(range(self.args.num_clients))
        
        # 重置服务器 / Reset server
        self.server.reset_round()
        
        # 存储客户端准确率 / Store client accuracies
        client_accuracies = {}
        
        # 是否显示详细信息 / Whether to show details
        show_details = (round_num % max(1, self.args.num_rounds // 10) == 0) or \
                      round_num == 1 or round_num == self.args.num_rounds
        
        if show_details:
            print(f"\n{'='*80}")
            print(f"Round {round_num}/{self.args.num_rounds}")
            print(f"{'='*80}")
        
        # =====================================================================
        # 步骤1: 客户端训练 / Step 1: Client Training
        # =====================================================================
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num}/{self.args.num_rounds}",
                            leave=False):
            client = self.clients[client_id]
            
            # 获取模型权重 / Get model weights
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
                epochs=self.args.local_epochs,
                lr=self.args.learning_rate
            )
            
            # 收集更新 / Collect updates
            self.server.collect_client_updates(client_id, updated_weights, train_info)
            
            # 记录准确率 / Record accuracy
            federated_acc = train_info['federated_accuracy']
            self.metrics_calculator.record_federated_accuracy(client_id, federated_acc)
            client_accuracies[client_id] = federated_acc
        
        # =====================================================================
        # 步骤2: CGSV贡献度计算（批量+归一化）/ Step 2: CGSV Calculation
        # =====================================================================
        normalized_contributions, raw_contributions = \
            self.server.calculate_all_contributions(round_num)
        
        # =====================================================================
        # 步骤3: 积分累加 / Step 3: Points Accumulation
        # =====================================================================
        all_active_points = {}
        for client_id, contribution in normalized_contributions.items():
            active_points = self.time_slice_manager.add_contribution_points(
                client_id, round_num, contribution
            )
            all_active_points[client_id] = active_points
        
        # =====================================================================
        # 步骤4: 基于相对排名更新会员等级 / Step 4: Update Membership by Ranking
        # =====================================================================
        new_levels = self.membership_system.update_all_memberships_by_ranking(
            all_active_points
        )
        
        # 更新客户端会员等级 / Update client membership levels
        for client_id, new_level in new_levels.items():
            self.clients[client_id].update_membership_level(new_level)
        
        # =====================================================================
        # 步骤5: 阶段性清理过期积分 / Step 5: Periodic Expiration
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
                    new_levels = self.membership_system.update_all_memberships_by_ranking(
                        updated_points
                    )
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
        round_time = time.time() - round_start
        
        # 打印详细信息 / Print details
        if show_details:
            if client_accuracies:
                accs = list(client_accuracies.values())
                print(f"Results:")
                print(f"  Avg Accuracy: {np.mean(accs):.4f}")
                print(f"  Max Accuracy: {np.max(accs):.4f}")
                print(f"  Min Accuracy: {np.min(accs):.4f}")
            
            print(f"  Time: {round_time:.2f}s")
            
            # 贡献度统计 / Contribution statistics
            if normalized_contributions:
                norm_vals = list(normalized_contributions.values())
                raw_vals = list(raw_contributions.values())
                print(f"  Raw CGSV - Mean: {np.mean(raw_vals):.4f}, Std: {np.std(raw_vals):.4f}")
                if self.args.use_relative_normalization:
                    print(f"  Normalized - Mean: {np.mean(norm_vals):.4f}, Std: {np.std(norm_vals):.4f}")
            
            # 积分和等级统计 / Points and level statistics
            if all_active_points:
                points_vals = list(all_active_points.values())
                print(f"  Active Points - Mean: {np.mean(points_vals):.2f}, Max: {np.max(points_vals):.2f}")
            
            # 会员等级分布 / Membership distribution
            if round_num % 10 == 0 or round_num == self.args.num_rounds:
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
            'membership_levels': {cid: self.clients[cid].membership_level 
                                 for cid in self.clients}
        }
        
        self.metrics_calculator.record_round(round_metrics)
        
        return round_metrics
    
    def run_experiment(self):
        """运行完整实验 / Run complete experiment"""
        print(f"\n{'='*80}")
        print(f"Starting Experiment: {self.experiment_name}")
        print(f"{'='*80}")
        
        # 1. 计算独立训练基准 / Compute standalone baselines
        self.compute_standalone_baselines()
        
        # 2. 联邦学习训练 / Federated learning training
        print(f"\n{'='*80}")
        print("Federated Learning Training")
        print(f"{'='*80}")
        print(f"Total Rounds: {self.args.num_rounds}")
        print(f"Local Epochs per Round: {self.args.local_epochs}")
        
        for round_num in range(1, self.args.num_rounds + 1):
            self.run_single_round(round_num)
        
        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        
        # 3. 计算最终指标（包含IPR）/ Calculate final metrics (including IPR)
        final_metrics = self.metrics_calculator.calculate_final_metrics()
        
        # 4. 打印摘要 / Print summary
        self.metrics_calculator.print_summary()
        
        # 5. 打印时间片统计 / Print time slice statistics
        self.time_slice_manager.print_summary(self.args.num_rounds)
        
        # 6. 打印贡献度统计 / Print contribution statistics
        self.server.print_contribution_summary()
        
        # 7. 打印会员分布 / Print membership distribution
        self.membership_system.print_membership_distribution()
        
        # 8. 生成可视化（包含IPR图表）/ Generate visualizations (including IPR plots)
        self._generate_visualizations(final_metrics)
        
        # 9. 保存结果 / Save results
        self._save_results(final_metrics)
        
        return final_metrics
    
    def _generate_visualizations(self, final_metrics: Dict):
        """生成可视化 / Generate visualizations"""
        print(f"\n{'='*80}")
        print("Generating Visualizations")
        print(f"{'='*80}")
        
        # 准备历史数据 / Prepare history data
        contributions_history = []
        raw_contributions_history = []
        for round_metric in self.metrics_calculator.round_metrics:
            contributions_history.append(
                round_metric.get('normalized_contributions', {})
            )
            raw_contributions_history.append(
                round_metric.get('raw_contributions', {})
            )
        
        metrics_history = {
            'rounds': list(range(1, len(self.metrics_calculator.avg_client_accuracies) + 1)),
            'avg_client_accuracy': self.metrics_calculator.avg_client_accuracies,
            'max_client_accuracy': self.metrics_calculator.max_client_accuracies,
            'time_per_round': self.metrics_calculator.time_consumptions,
            'contributions': contributions_history,
            'raw_contributions': raw_contributions_history
        }
        
        # 生成所有图表 / Generate all plots
        self.visualizer.generate_all_plots(
            final_metrics,
            metrics_history,
            self.experiment_name
        )
        
        print("✓ All visualizations generated")
    
    def _save_results(self, final_metrics: Dict):
        """保存结果 / Save results"""
        print(f"\n{'='*80}")
        print("Saving Results")
        print(f"{'='*80}")
        
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存完整结果 / Save complete results
        import json
        results_path = os.path.join(results_dir, f"{self.experiment_name}_results.json")
        
        save_data = {
            'experiment_name': self.experiment_name,
            'configuration': {
                'dataset': self.args.dataset,
                'num_clients': self.args.num_clients,
                'num_rounds': self.args.num_rounds,
                'distribution': self.args.distribution,
                'local_epochs': self.args.local_epochs,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'standalone_epochs': self.args.standalone_epochs,
                'time_slice_type': self.args.time_slice_type,
                'rounds_per_slice': self.args.rounds_per_slice,
                'use_relative_normalization': self.args.use_relative_normalization,
                'contribution_method': 'CGSV',
                'membership_method': 'Relative Ranking',
                'seed': self.args.seed
            },
            'final_metrics': final_metrics,
            'round_metrics': self.metrics_calculator.round_metrics,
            'membership_statistics': self.membership_system.get_membership_statistics(),
            'contribution_statistics': self.server.get_contribution_statistics()
        }
        
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {results_path}")


def parse_args():
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Federated Learning with CGSV + Relative Ranking + IPR'
    )
    
    # 数据集参数 / Dataset parameters
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'fashion-mnist'],
                       help='Dataset name (case-insensitive)')
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients')
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non-iid'],
                       help='Data distribution type')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-IID distribution')
    
    # 训练参数 / Training parameters
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Number of local epochs per round')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for local training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--standalone_epochs', type=int, default=20,
                       help='Epochs for standalone training baseline')
    
    # 激励机制参数 / Incentive mechanism parameters
    parser.add_argument('--time_slice_type', type=str, default='rounds',
                       choices=['rounds', 'time'],
                       help='Time slice type')
    parser.add_argument('--rounds_per_slice', type=int, default=10,
                       help='Rounds per time slice')
    parser.add_argument('--use_relative_normalization', type=bool, default=True,
                       help='Use relative normalization for CGSV contributions')
    
    # 其他参数 / Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 统一数据集名称为小写（兼容性处理）
    # Normalize dataset name to lowercase (compatibility)
    args.dataset = args.dataset.lower()
    
    return args


def main():
    """主函数 / Main function"""
    # 解析参数 / Parse arguments
    args = parse_args()
    
    # 设置设备 / Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 更新全局设备 / Update global device
    import config
    config.DEVICE = device
    
    # 创建并运行实验 / Create and run experiment
    experiment = FederatedLearningExperiment(args)
    final_metrics = experiment.run_experiment()
    
    print(f"\n{'='*80}")
    print("Experiment Completed Successfully!")
    print(f"{'='*80}")
    print(f"Experiment Name: {experiment.experiment_name}")
    print(f"Final IPR: {final_metrics['ipr']['final_ipr']:.4f} "
          f"({final_metrics['ipr']['ipr_percentage']:.2f}%)")
    print(f"Final Avg Accuracy: {final_metrics['client_accuracy']['avg_final']:.4f}")
    print(f"PCC: {final_metrics['pcc']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()