"""
main.py (Updated for UPSM Implementation)
主程序 - 实现PDF文档中的UPSM统一概率采样机制
Main Program - Implements UPSM from PDF

用法 / Usage:
    python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5
    python main.py --dataset mnist --distribution iid --num_rounds 100
"""

import torch
import numpy as np
import random
import argparse
import os
import sys
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FederatedConfig, IncentiveConfig, DatasetConfig, DEVICE
from datasets.data_loader import FederatedDataLoader
from models.cnn_model import ModelFactory
from federated.server import FederatedServer
from federated.client import FederatedClient
from incentive.time_slice import TimeSliceManager
from incentive.membership import MembershipSystem
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer


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


class FederatedLearningExperiment:
    """
    联邦学习实验 - UPSM实现 / Federated Learning Experiment - UPSM Implementation
    """
    
    def __init__(self, args):
        """初始化实验 / Initialize experiment"""
        self.args = args
        set_seed(args.seed)
        
        self.device = DEVICE
        self.experiment_name = self._generate_experiment_name()
        
        print(f"\n{'='*80}")
        print(f"Federated Learning with UPSM")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Distribution: {args.distribution}")
        print(f"Clients: {args.num_clients}")
        print(f"Rounds: {args.num_rounds}")
        print(f"Time Slice: τ={args.rounds_per_slice}, W={IncentiveConfig.POINTS_VALIDITY_SLICES}")
        print(f"Device: {self.device}")
        print(f"{'='*80}")
        
        self._initialize_components()
        
    def _generate_experiment_name(self) -> str:
        """生成实验名称 / Generate experiment name"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dist_suffix = f"_a{self.args.alpha}" if self.args.distribution == "non-iid-dir" else ""
        return f"UPSM_{self.args.dataset}_{self.args.distribution}{dist_suffix}_" \
               f"c{self.args.num_clients}_r{self.args.num_rounds}_{timestamp}"
    
    def _initialize_components(self):
        """初始化所有组件 / Initialize all components"""
        print("\nInitializing components...")
        
        # 1. 数据 / Data
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
        
        if self.args.dataset == "sst":
            self.model = ModelFactory.create_model(
                self.args.dataset,
                num_classes=num_classes,
                vocab_size=DatasetConfig.SST_VOCAB_SIZE,
                embedding_dim=DatasetConfig.SST_EMBEDDING_DIM,
                max_seq_length=DatasetConfig.SST_MAX_SEQ_LENGTH
            )
        else:
            self.model = ModelFactory.create_model(
                self.args.dataset,
                num_classes=num_classes,
                input_channels=input_channels
            )
        
        # 3. 服务器 / Server
        print("  [3/6] Initializing server with UPSM...")
        self.server = FederatedServer(self.model, self.device)
        
        # 4. 客户端 / Clients
        print("  [4/6] Creating clients...")
        self.clients = {}
        for client_id in tqdm(range(self.args.num_clients), desc="    Creating", leave=False):
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
        
        # 5. 激励系统 / Incentive system
        print("  [5/6] Initializing incentive system...")
        self.time_slice_manager = TimeSliceManager(
            slice_type="rounds",
            rounds_per_slice=self.args.rounds_per_slice,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        self.membership_system = MembershipSystem(
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS,
            ranking_percentiles=IncentiveConfig.LEVEL_PERCENTILES
        )
        
        for client_id in range(self.args.num_clients):
            self.membership_system.initialize_client(client_id)
        
        # 6. 指标 / Metrics
        print("  [6/6] Initializing metrics...")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(output_dir="outputs/figures")
        
        print("✓ All components initialized")
    
    def compute_standalone_baselines(self):
        """计算独立训练基准 / Compute standalone baselines"""
        print(f"\n{'='*80}")
        print(f"Computing Standalone Baselines ({self.args.standalone_epochs} epochs)")
        print(f"{'='*80}")
        
        for client_id, client in tqdm(self.clients.items(), desc="Standalone training"):
            standalone_acc, _ = client.train_standalone(epochs=self.args.standalone_epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        print("✓ Standalone baselines computed")
    
    def run_single_round(self, round_num: int) -> dict:
        """运行单轮训练 / Run single round"""
        round_start = time.time()
        
        selected_clients = list(range(self.args.num_clients))
        self.server.reset_round()
        client_accuracies = {}
        
        show_details = (round_num % max(1, self.args.num_rounds // 10) == 0) or \
                      round_num == 1 or round_num == self.args.num_rounds
        
        if show_details:
            print(f"\n{'='*80}")
            print(f"Round {round_num}/{self.args.num_rounds}")
            print(f"{'='*80}")
        
        # 客户端训练 / Client training
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num}/{self.args.num_rounds}",
                            leave=False):
            client = self.clients[client_id]
            
            if round_num == 1:
                model_weights = self.server.get_global_model_weights()
            else:
                if hasattr(self, 'personalized_models') and client_id in self.personalized_models:
                    model_weights = self.personalized_models[client_id]
                else:
                    model_weights = self.server.get_global_model_weights()
            
            updated_weights, train_info = client.train_federated(
                global_weights=model_weights,
                epochs=self.args.local_epochs,
                lr=self.args.learning_rate
            )
            
            self.server.collect_client_updates(client_id, updated_weights, train_info)
            
            federated_acc = train_info['federated_accuracy']
            self.metrics_calculator.record_federated_accuracy(client_id, federated_acc)
            client_accuracies[client_id] = federated_acc
        
        # 计算贡献度 / Calculate contributions
        contributions = self.server.calculate_all_contributions(round_num)
        
        # 累计积分 / Accumulate points
        all_active_points = {}
        for client_id, contribution in contributions.items():
            active_points = self.time_slice_manager.add_contribution_points(
                client_id, round_num, contribution
            )
            all_active_points[client_id] = active_points
        
        # 更新会员等级 / Update membership
        new_levels = self.membership_system.update_all_memberships_by_ranking(all_active_points)
        for client_id, new_level in new_levels.items():
            self.clients[client_id].update_membership_level(new_level)
        
        # 清理过期积分 / Clean expired points
        current_slice = self.time_slice_manager.get_current_slice(round_num)
        if round_num > 1:
            prev_slice = self.time_slice_manager.get_current_slice(round_num - 1)
            if current_slice != prev_slice:
                cleaned = self.time_slice_manager.clean_expired_points(round_num)
                if cleaned:
                    updated_points = self.time_slice_manager.get_all_client_active_points(round_num)
                    new_levels = self.membership_system.update_all_memberships_by_ranking(updated_points)
                    for client_id, new_level in new_levels.items():
                        self.clients[client_id].update_membership_level(new_level)
        
        # UPSM分发 / UPSM distribution
        self.personalized_models = self.server.distribute_personalized_models(new_levels)
        
        # 更新全局模型 / Update global model
        self.server.update_global_model()
        
        round_time = time.time() - round_start
        
        # 打印详情 / Print details
        if show_details:
            if client_accuracies:
                accs = list(client_accuracies.values())
                print(f"Results:")
                print(f"  Avg Accuracy: {np.mean(accs):.4f}")
                print(f"  Max Accuracy: {np.max(accs):.4f}")
                print(f"  Min Accuracy: {np.min(accs):.4f}")
            print(f"  Time: {round_time:.2f}s")
            
            if round_num % 10 == 0 or round_num == self.args.num_rounds:
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
    
    def run_experiment(self):
        """运行完整实验 / Run complete experiment"""
        print(f"\n{'='*80}")
        print(f"Starting: {self.experiment_name}")
        print(f"{'='*80}")
        
        # 独立训练基准 / Standalone baselines
        self.compute_standalone_baselines()
        
        # 联邦训练 / Federated training
        print(f"\n{'='*80}")
        print("Federated Learning Training")
        print(f"{'='*80}")
        
        for round_num in range(1, self.args.num_rounds + 1):
            self.run_single_round(round_num)
        
        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        
        # 最终指标 / Final metrics
        final_metrics = self.metrics_calculator.calculate_final_metrics()
        
        # 打印摘要 / Print summaries
        self.metrics_calculator.print_summary()
        self.time_slice_manager.print_summary(self.args.num_rounds)
        self.server.print_contribution_summary()
        self.membership_system.print_membership_distribution()
        
        # 可视化 / Visualizations
        self._generate_visualizations(final_metrics)
        
        # 保存结果 / Save results
        self._save_results(final_metrics)
        
        return final_metrics
    
    def _generate_visualizations(self, final_metrics):
        """生成可视化 / Generate visualizations"""
        print(f"\n{'='*80}")
        print("Generating Visualizations")
        print(f"{'='*80}")
        
        contributions_history = [rm.get('contributions', {}) for rm in self.metrics_calculator.round_metrics]
        
        metrics_history = {
            'rounds': list(range(1, len(self.metrics_calculator.avg_client_accuracies) + 1)),
            'avg_client_accuracy': self.metrics_calculator.avg_client_accuracies,
            'max_client_accuracy': self.metrics_calculator.max_client_accuracies,
            'time_per_round': self.metrics_calculator.time_consumptions,
            'contributions': contributions_history,
            'raw_contributions': contributions_history
        }
        
        self.visualizer.generate_all_plots(final_metrics, metrics_history, self.experiment_name)
        print("✓ Visualizations generated")
    
    def _save_results(self, final_metrics):
        """保存结果 / Save results"""
        import json
        
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"{self.experiment_name}_results.json")
        
        save_data = {
            'experiment_name': self.experiment_name,
            'configuration': {
                'dataset': self.args.dataset,
                'num_clients': self.args.num_clients,
                'num_rounds': self.args.num_rounds,
                'distribution': self.args.distribution,
                'alpha': self.args.alpha,
                'rounds_per_slice': self.args.rounds_per_slice,
                'local_epochs': self.args.local_epochs,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'standalone_epochs': self.args.standalone_epochs,
                'seed': self.args.seed
            },
            'upsm_config': {
                'access_ratios': IncentiveConfig.LEVEL_ACCESS_RATIOS,
                'selection_bias': IncentiveConfig.LEVEL_SELECTION_BIAS,
                'level_percentiles': IncentiveConfig.LEVEL_PERCENTILES
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
        description='Federated Learning with UPSM (Unified Probabilistic Sampling Mechanism)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 使用示例:
  # IID distribution
  python main.py --dataset mnist --distribution iid
  
  # Non-IID with Dirichlet
  python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5
  
  # Larger experiment
  python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.1 \\
                 --num_clients 100 --num_rounds 100
        """
    )
    
    # 数据集 / Dataset
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'sst'],
                       help='Dataset name')
    
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients')
    
    # 分布 / Distribution
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non-iid-dir'],
                       help='Data distribution type')
    
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-iid-dir')
    
    # 训练 / Training
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of communication rounds')
    
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local epochs per round')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    
    parser.add_argument('--standalone_epochs', type=int, default=20,
                       help='Standalone training epochs')
    
    # 时间片 / Time slice
    parser.add_argument('--rounds_per_slice', type=int, default=5,
                       help='Rounds per time slice (τ)')
    
    # 其他 / Others
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """主函数 / Main function"""
    args = parse_args()
    
    # 设置设备 / Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import config
    config.DEVICE = device
    
    # 运行实验 / Run experiment
    experiment = FederatedLearningExperiment(args)
    final_metrics = experiment.run_experiment()
    
    # 打印最终结果 / Print final results
    print(f"\n{'='*80}")
    print("Experiment Completed!")
    print(f"{'='*80}")
    print(f"Experiment: {experiment.experiment_name}")
    print(f"\nKey Results:")
    print(f"  Final Avg Accuracy: {final_metrics['client_accuracy']['avg_final']:.4f}")
    print(f"  PCC: {final_metrics['pcc']:.4f}")
    print(f"  IPR: {final_metrics['ipr']['final_ipr']:.4f} ({final_metrics['ipr']['ipr_percentage']:.2f}%)")
    print(f"  Total Time: {final_metrics['time_consumption']['total']:.2f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()