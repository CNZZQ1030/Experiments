"""
main.py (Extended Version)
主程序 - 扩展版本 / Main Program - Extended Version

支持多种数据集和分布类型的联邦学习实验
Supports federated learning experiments with multiple datasets and distribution types

支持的数据集 / Supported Datasets:
- mnist, fashion-mnist, cifar10, cifar100, sst

支持的分布类型 / Supported Distribution Types:
- iid: 独立同分布 / IID
- non-iid-dir: Dirichlet分布 / Dirichlet (quantity skew)
- non-iid-size: 数据量不平衡 / Imbalanced dataset size
- non-iid-class: 类别数不平衡 / Imbalanced class number

用法示例 / Usage Examples:
    # MNIST with IID distribution
    python main.py --dataset mnist --distribution iid
    
    # CIFAR-100 with Dirichlet Non-IID
    python main.py --dataset cifar100 --distribution non-iid-dir --alpha 0.5
    
    # SST with imbalanced size
    python main.py --dataset sst --distribution non-iid-size --size_ratio 5.0
    
    # CIFAR-10 with imbalanced class number
    python main.py --dataset cifar10 --distribution non-iid-class --min_classes 2 --max_classes 5
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
    联邦学习实验类 - 扩展版本 / Federated Learning Experiment Class - Extended Version
    完整集成CGSV + 相对排名 + IPR，支持多种数据集和分布
    Complete integration of CGSV + Relative Ranking + IPR with multiple datasets and distributions
    """
    
    def __init__(self, args):
        """
        初始化实验 / Initialize experiment
        
        Args:
            args: 命令行参数 / Command line arguments
        """
        self.args = args
        
        # 统一数据集名称为小写 / Normalize dataset name to lowercase
        self.args.dataset = self.args.dataset.lower()
        
        # 设置随机种子 / Set random seed
        set_seed(args.seed)
        
        # 设备 / Device
        self.device = DEVICE
        
        # 实验名称 / Experiment name
        self.experiment_name = self._generate_experiment_name()
        
        print(f"\n{'='*80}")
        print(f"Federated Learning Experiment - Extended Version")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Distribution: {args.distribution}")
        print(f"Clients: {args.num_clients}")
        print(f"Rounds: {args.num_rounds}")
        print(f"Device: {self.device}")
        
        # 显示分布特定参数 / Show distribution-specific parameters
        if args.distribution == "non-iid-dir":
            print(f"  Alpha: {args.alpha}")
        elif args.distribution == "non-iid-size":
            print(f"  Size Imbalance Ratio: {args.size_ratio}")
        elif args.distribution == "non-iid-class":
            print(f"  Classes per Client: {args.min_classes}-{args.max_classes}")
        
        print(f"{'='*80}")
        
        # 初始化组件 / Initialize components
        self._initialize_components()
        
    def _generate_experiment_name(self) -> str:
        """生成实验名称 / Generate experiment name"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dist_suffix = ""
        if self.args.distribution == "non-iid-dir":
            dist_suffix = f"_a{self.args.alpha}"
        elif self.args.distribution == "non-iid-size":
            dist_suffix = f"_r{self.args.size_ratio}"
        elif self.args.distribution == "non-iid-class":
            dist_suffix = f"_c{self.args.min_classes}-{self.args.max_classes}"
        
        name = f"{self.args.dataset}_{self.args.distribution}{dist_suffix}_" \
               f"c{self.args.num_clients}_r{self.args.num_rounds}_{timestamp}"
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
            alpha=self.args.alpha,
            size_imbalance_ratio=self.args.size_ratio,
            min_classes_per_client=self.args.min_classes,
            max_classes_per_client=self.args.max_classes
        )
        
        # 2. 模型 / Model
        print("  [2/6] Creating model...")
        num_classes = DatasetConfig.NUM_CLASSES[self.args.dataset]
        input_channels = DatasetConfig.INPUT_SHAPE[self.args.dataset][0]
        
        # 根据数据集类型传递不同参数 / Pass different params based on dataset type
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
        print("  [3/6] Initializing server...")
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
        
        # 5. 激励系统 / Incentive system
        print("  [5/6] Initializing incentive system...")
        self.time_slice_manager = TimeSliceManager(
            slice_type=self.args.time_slice_type,
            rounds_per_slice=self.args.rounds_per_slice,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        self.membership_system = MembershipSystem(
            level_multipliers=IncentiveConfig.LEVEL_MULTIPLIERS
        )
        
        for client_id in range(self.args.num_clients):
            self.membership_system.initialize_client(client_id)
        
        # 6. 评估和可视化 / Evaluation and visualization
        print("  [6/6] Initializing metrics and visualization...")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(output_dir="outputs/figures")
        
        print("✓ All components initialized")
    
    def compute_standalone_baselines(self):
        """计算独立训练基准 / Compute standalone baselines"""
        print(f"\n{'='*80}")
        print("Computing Standalone Baselines")
        print(f"{'='*80}")
        print(f"Each client trains independently for {self.args.standalone_epochs} epochs")
        
        for client_id, client in tqdm(self.clients.items(), desc="Standalone training"):
            standalone_acc, _ = client.train_standalone(epochs=self.args.standalone_epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        print("✓ Standalone baselines computed")
    
    def run_single_round(self, round_num: int) -> Dict:
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
        
        # CGSV贡献度计算 / CGSV calculation
        normalized_contributions, raw_contributions = \
            self.server.calculate_all_contributions(round_num)
        
        # 积分累加 / Points accumulation
        all_active_points = {}
        for client_id, contribution in normalized_contributions.items():
            active_points = self.time_slice_manager.add_contribution_points(
                client_id, round_num, contribution
            )
            all_active_points[client_id] = active_points
        
        # 更新会员等级 / Update membership
        new_levels = self.membership_system.update_all_memberships_by_ranking(all_active_points)
        for client_id, new_level in new_levels.items():
            self.clients[client_id].update_membership_level(new_level)
        
        # 阶段性清理 / Periodic cleanup
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
        
        # 分发个性化模型 / Distribute personalized models
        self.personalized_models = self.server.distribute_personalized_models(round_num)
        
        # 更新全局模型 / Update global model
        self.server.update_global_model()
        
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
            'membership_levels': {cid: self.clients[cid].membership_level for cid in self.clients}
        }
        
        self.metrics_calculator.record_round(round_metrics)
        
        return round_metrics
    
    def run_experiment(self):
        """运行完整实验 / Run complete experiment"""
        print(f"\n{'='*80}")
        print(f"Starting Experiment: {self.experiment_name}")
        print(f"{'='*80}")
        
        # 计算独立训练基准 / Compute baselines
        self.compute_standalone_baselines()
        
        # 联邦学习训练 / Federated training
        print(f"\n{'='*80}")
        print("Federated Learning Training")
        print(f"{'='*80}")
        
        for round_num in range(1, self.args.num_rounds + 1):
            self.run_single_round(round_num)
        
        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        
        # 计算最终指标 / Calculate final metrics
        final_metrics = self.metrics_calculator.calculate_final_metrics()
        
        # 打印摘要 / Print summary
        self.metrics_calculator.print_summary()
        self.time_slice_manager.print_summary(self.args.num_rounds)
        self.server.print_contribution_summary()
        self.membership_system.print_membership_distribution()
        
        # 生成可视化 / Generate visualizations
        self._generate_visualizations(final_metrics)
        
        # 保存结果 / Save results
        self._save_results(final_metrics)
        
        return final_metrics
    
    def _generate_visualizations(self, final_metrics: Dict):
        """生成可视化 / Generate visualizations"""
        print(f"\n{'='*80}")
        print("Generating Visualizations")
        print(f"{'='*80}")
        
        contributions_history = []
        raw_contributions_history = []
        for round_metric in self.metrics_calculator.round_metrics:
            contributions_history.append(round_metric.get('normalized_contributions', {}))
            raw_contributions_history.append(round_metric.get('raw_contributions', {}))
        
        metrics_history = {
            'rounds': list(range(1, len(self.metrics_calculator.avg_client_accuracies) + 1)),
            'avg_client_accuracy': self.metrics_calculator.avg_client_accuracies,
            'max_client_accuracy': self.metrics_calculator.max_client_accuracies,
            'time_per_round': self.metrics_calculator.time_consumptions,
            'contributions': contributions_history,
            'raw_contributions': raw_contributions_history
        }
        
        self.visualizer.generate_all_plots(final_metrics, metrics_history, self.experiment_name)
        print("✓ All visualizations generated")
    
    def _save_results(self, final_metrics: Dict):
        """保存结果 / Save results"""
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        import json
        results_path = os.path.join(results_dir, f"{self.experiment_name}_results.json")
        
        save_data = {
            'experiment_name': self.experiment_name,
            'configuration': {
                'dataset': self.args.dataset,
                'num_clients': self.args.num_clients,
                'num_rounds': self.args.num_rounds,
                'distribution': self.args.distribution,
                'alpha': self.args.alpha,
                'size_ratio': self.args.size_ratio,
                'min_classes': self.args.min_classes,
                'max_classes': self.args.max_classes,
                'local_epochs': self.args.local_epochs,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'standalone_epochs': self.args.standalone_epochs,
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
    """
    解析命令行参数 / Parse command line arguments
    
    支持多种数据集和分布类型
    Supports multiple datasets and distribution types
    """
    parser = argparse.ArgumentParser(
        description='Federated Learning with Extended Datasets and Distributions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 使用示例:
  # MNIST with IID distribution / MNIST IID分布
  python main.py --dataset mnist --distribution iid
  
  # CIFAR-100 with Dirichlet Non-IID / CIFAR-100 Dirichlet非独立同分布
  python main.py --dataset cifar100 --distribution non-iid-dir --alpha 0.5
  
  # SST with imbalanced size / SST 数据量不平衡
  python main.py --dataset sst --distribution non-iid-size --size_ratio 5.0
  
  # CIFAR-10 with imbalanced class / CIFAR-10 类别数不平衡
  python main.py --dataset cifar10 --distribution non-iid-class --min_classes 2 --max_classes 5
        """
    )
    
    # ===== 数据集参数 / Dataset parameters =====
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'sst'],
                       help='Dataset name / 数据集名称')
    
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients / 客户端数量')
    
    # ===== 分布参数 / Distribution parameters =====
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non-iid-dir', 'non-iid-size', 'non-iid-class'],
                       help='Data distribution type / 数据分布类型')
    
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-iid-dir distribution / Dirichlet参数')
    
    parser.add_argument('--size_ratio', type=float, default=5.0,
                       help='Max/min data ratio for non-iid-size distribution / 数据量不平衡比例')
    
    parser.add_argument('--min_classes', type=int, default=2,
                       help='Min classes per client for non-iid-class / 每客户端最少类别数')
    
    parser.add_argument('--max_classes', type=int, default=5,
                       help='Max classes per client for non-iid-class / 每客户端最多类别数')
    
    # ===== 训练参数 / Training parameters =====
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of communication rounds / 通信轮次')
    
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Number of local epochs per round / 每轮本地训练轮次')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for local training / 本地训练批次大小')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate / 学习率')
    
    parser.add_argument('--standalone_epochs', type=int, default=20,
                       help='Epochs for standalone training baseline / 独立训练轮次')
    
    # ===== 激励机制参数 / Incentive mechanism parameters =====
    parser.add_argument('--time_slice_type', type=str, default='rounds',
                       choices=['rounds', 'time'],
                       help='Time slice type / 时间片类型')
    
    parser.add_argument('--rounds_per_slice', type=int, default=10,
                       help='Rounds per time slice / 每时间片轮次数')
    
    parser.add_argument('--use_relative_normalization', type=bool, default=True,
                       help='Use relative normalization for CGSV / 使用相对归一化')
    
    # ===== 其他参数 / Other parameters =====
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed / 随机种子')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use / 计算设备')
    
    args = parser.parse_args()
    
    # 统一数据集名称为小写 / Normalize dataset name
    args.dataset = args.dataset.lower()
    
    return args


def print_usage_examples():
    """打印使用示例 / Print usage examples"""
    examples = """
================================================================================
Federated Learning Experiment - Usage Examples / 使用示例
================================================================================

1. Basic IID Experiments / 基础IID实验:
   python main.py --dataset mnist --distribution iid --num_clients 100 --num_rounds 50
   python main.py --dataset cifar10 --distribution iid --num_clients 100 --num_rounds 100
   python main.py --dataset cifar100 --distribution iid --num_clients 100 --num_rounds 150

2. Dirichlet Non-IID (Quantity Skew) / Dirichlet非独立同分布:
   python main.py --dataset mnist --distribution non-iid-dir --alpha 0.1
   python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5
   python main.py --dataset cifar100 --distribution non-iid-dir --alpha 1.0

3. Imbalanced Dataset Size / 数据量不平衡:
   python main.py --dataset mnist --distribution non-iid-size --size_ratio 5.0
   python main.py --dataset cifar10 --distribution non-iid-size --size_ratio 10.0

4. Imbalanced Class Number / 类别数不平衡:
   python main.py --dataset mnist --distribution non-iid-class --min_classes 1 --max_classes 3
   python main.py --dataset cifar100 --distribution non-iid-class --min_classes 5 --max_classes 20

5. SST Text Classification / SST文本分类:
   python main.py --dataset sst --distribution iid --num_rounds 30
   python main.py --dataset sst --distribution non-iid-dir --alpha 0.5

6. Full Configuration Example / 完整配置示例:
   python main.py \\
       --dataset cifar10 \\
       --distribution non-iid-dir \\
       --alpha 0.5 \\
       --num_clients 100 \\
       --num_rounds 100 \\
       --local_epochs 5 \\
       --batch_size 32 \\
       --learning_rate 0.01 \\
       --standalone_epochs 20 \\
       --seed 42

================================================================================
"""
    print(examples)


def main():
    """主函数 / Main function"""
    # 解析参数 / Parse arguments
    args = parse_args()
    
    # 如果没有提供任何参数，打印示例 / Print examples if no args
    if len(sys.argv) == 1:
        print_usage_examples()
        print("Running with default configuration: MNIST + IID\n")
    
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
    print(f"Dataset: {args.dataset}")
    print(f"Distribution: {args.distribution}")
    print(f"Final IPR: {final_metrics['ipr']['final_ipr']:.4f} "
          f"({final_metrics['ipr']['ipr_percentage']:.2f}%)")
    print(f"Final Avg Accuracy: {final_metrics['client_accuracy']['avg_final']:.4f}")
    print(f"PCC: {final_metrics['pcc']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()