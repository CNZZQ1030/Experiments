"""
main.py - 重构版本
主程序 - 基于梯度稀疏化的联邦学习激励机制
Main Program - Gradient Sparsification-based Federated Learning Incentive Mechanism

核心流程 / Core Workflow:
1. 服务器收集客户端梯度 Δw_i
2. 聚合得到全局梯度 Δw_global = Σ(n_i/n * Δw_i)
3. 对全局梯度进行差异化稀疏 sparse(Δw_global)
4. 客户端应用稀疏梯度 w_local^(t+1) = w_local^(t) + sparse(Δw_global)

使用方法 / Usage:
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
import json
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FederatedConfig, IncentiveConfig, DatasetConfig, DEVICE
from datasets.data_loader import FederatedDataLoader
from models.cnn_model import ModelFactory
from incentive.time_slice import TimeSliceManager
from incentive.membership import MembershipSystem
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer

# 导入重构的服务器和客户端
from federated.server import FederatedServerWithGradientSparsification
from federated.client import FederatedClient


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GradientSparsificationFederatedLearning:
    """
    基于梯度稀疏化的联邦学习实验
    
    核心创新 / Core Innovation:
    - 聚合全局梯度而非模型权重
    - 对梯度进行差异化稀疏
    - 客户端累积应用稀疏梯度到本地模型
    - 保持客户端本地个性化
    """
    
    def __init__(self, args):
        """初始化实验"""
        self.args = args
        set_seed(args.seed)
        
        self.device = DEVICE
        self.experiment_name = self._generate_experiment_name()
        
        print(f"\n{'='*80}")
        print(f"Federated Learning with Gradient Sparsification Incentive")
        print(f"联邦学习 - 基于梯度稀疏化的激励机制")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Distribution: {args.distribution}")
        print(f"Clients: {args.num_clients}")
        print(f"Rounds: {args.num_rounds}")
        print(f"Device: {self.device}")
        print(f"\n✨ Gradient Sparsification Configuration:")
        print(f"  Mode: {args.sparsification_mode}")
        print(f"  Lambda: {args.lambda_coef}")
        print(f"  Min Keep Ratio: {args.min_keep_ratio}")
        print(f"  Gradient Application LR: {args.gradient_lr}")
        print(f"\n📊 Expected Sparsity Ranges:")
        for level, (min_s, max_s) in IncentiveConfig.LEVEL_SPARSITY_RANGES.items():
            keep_min, keep_max = 1.0 - max_s, 1.0 - min_s
            print(f"    {level.capitalize()}: Keep [{keep_min:.2f}, {keep_max:.2f}] "
                  f"(Sparse [{min_s:.2f}, {max_s:.2f}])")
        print(f"{'='*80}")
        
        # 更新配置
        IncentiveConfig.SPARSIFICATION_MODE = args.sparsification_mode
        IncentiveConfig.LAMBDA = args.lambda_coef
        IncentiveConfig.MIN_KEEP_RATIO = args.min_keep_ratio
        
        self._initialize_components()
    
    def _generate_experiment_name(self) -> str:
        """生成实验名称"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dist_suffix = f"_a{self.args.alpha}" if self.args.distribution == "non-iid-dir" else ""
        sparse_suffix = f"_GradSparse_{self.args.sparsification_mode}_l{self.args.lambda_coef}"
        return f"{self.args.dataset}_{self.args.distribution}{dist_suffix}" \
               f"_c{self.args.num_clients}_r{self.args.num_rounds}{sparse_suffix}_{timestamp}"
    
    def _initialize_components(self):
        """初始化所有组件"""
        print("\nInitializing components...")
        
        # 1. 数据加载
        print("  [1/6] Loading data...")
        self.data_loader = FederatedDataLoader(
            dataset_name=self.args.dataset,
            num_clients=self.args.num_clients,
            batch_size=self.args.batch_size,
            distribution=self.args.distribution,
            alpha=self.args.alpha
        )
        
        # 2. 模型创建
        print("  [2/6] Creating model...")
        num_classes = DatasetConfig.NUM_CLASSES[self.args.dataset]
        input_channels = DatasetConfig.INPUT_SHAPE[self.args.dataset][0]
        
        self.model = ModelFactory.create_model(
            self.args.dataset,
            num_classes=num_classes,
            input_channels=input_channels
        )
        
        # 3. 服务器初始化（梯度稀疏化版本）
        print("  [3/6] Initializing server with gradient sparsification...")
        self.server = FederatedServerWithGradientSparsification(self.model, self.device)
        
        # 4. 客户端创建
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
        
        # 5. 激励系统
        print("  [5/6] Initializing incentive system...")
        self.time_slice_manager = TimeSliceManager(
            slice_type="rounds",
            rounds_per_slice=self.args.rounds_per_slice,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        self.membership_system = MembershipSystem(
            ranking_percentiles=IncentiveConfig.LEVEL_PERCENTILES
        )
        
        for client_id in range(self.args.num_clients):
            self.membership_system.initialize_client(client_id)
        
        # 6. 指标系统
        print("  [6/6] Initializing metrics...")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(output_dir="outputs/figures")
        
        print("✓ All components initialized")
    
    def compute_standalone_baselines(self):
        """计算独立训练基准"""
        print(f"\n{'='*80}")
        print(f"Computing Standalone Baselines ({self.args.standalone_epochs} epochs)")
        print(f"{'='*80}")
        
        for client_id, client in tqdm(self.clients.items(), desc="Standalone training"):
            standalone_acc, _ = client.train_standalone(epochs=self.args.standalone_epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        print("✓ Standalone baselines computed")
    
    def run_single_round(self, round_num: int) -> dict:
        """
        运行单轮训练 - 梯度稀疏化版本
        
        核心流程 / Core Workflow:
        1. 客户端本地训练，上传训练后的权重
        2. 服务器计算客户端梯度：Δw_i = w_i^new - w_i^old
        3. 聚合全局梯度：Δw_global = Σ(n_i/n * Δw_i)
        4. 计算贡献度和更新会员等级
        5. 对全局梯度进行差异化稀疏
        6. 客户端应用稀疏梯度：w_local = w_local + lr * sparse(Δw_global)
        """
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
        
        # ========== 步骤1: 客户端本地训练 ==========
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num} - Training",
                            leave=False):
            client = self.clients[client_id]
            
            # 第一轮需要初始化本地模型
            if round_num == 1:
                global_weights = self.server.get_global_model_weights()
            else:
                global_weights = None  # 后续轮次使用本地模型
            
            # 本地训练（客户端保持本地模型状态）
            updated_weights, train_info = client.train_federated(
                global_weights=global_weights,
                epochs=self.args.local_epochs,
                lr=self.args.learning_rate
            )
            
            # 服务器收集更新（内部计算梯度）
            self.server.collect_client_updates(client_id, updated_weights, train_info)
            
            # 记录准确率
            federated_acc = train_info['federated_accuracy']
            self.metrics_calculator.record_federated_accuracy(client_id, federated_acc)
            client_accuracies[client_id] = federated_acc
        
        # ========== 步骤2: 聚合全局梯度 ==========
        self.server.update_global_model()
        
        # ========== 步骤3: 计算贡献度 ==========
        contributions = self.server.calculate_all_contributions(round_num)
        
        # ========== 步骤4: 时间片积分和会员等级 ==========
        all_active_points = {}
        for client_id, contribution in contributions.items():
            active_points = self.time_slice_manager.add_contribution_points(
                client_id, round_num, contribution
            )
            all_active_points[client_id] = active_points
        
        # 更新会员等级
        new_levels = self.membership_system.update_all_memberships_by_ranking(all_active_points)
        for client_id, new_level in new_levels.items():
            self.clients[client_id].update_membership_level(new_level)
        
        # 清理过期积分
        current_slice = self.time_slice_manager.get_current_slice(round_num)
        if round_num > 1:
            prev_slice = self.time_slice_manager.get_current_slice(round_num - 1)
            if current_slice != prev_slice:
                cleaned = self.time_slice_manager.clean_expired_points(round_num)
                if cleaned and show_details:
                    print(f"  Time slice changed: {prev_slice} → {current_slice}")
                
                # 重新计算等级
                updated_points = self.time_slice_manager.get_all_client_active_points(round_num)
                new_levels = self.membership_system.update_all_memberships_by_ranking(updated_points)
                for client_id, new_level in new_levels.items():
                    self.clients[client_id].update_membership_level(new_level)
        
        # ========== 步骤5: 分发稀疏化梯度 ==========
        sparsified_gradients = self.server.distribute_sparsified_gradients(new_levels)
        
        # ========== 步骤6: 客户端应用稀疏梯度 ==========
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num} - Applying gradients",
                            leave=False):
            if client_id in sparsified_gradients:
                sparse_gradient = sparsified_gradients[client_id]
                self.clients[client_id].apply_gradient_update(
                    sparse_gradient, 
                    learning_rate=self.args.gradient_lr
                )
        
        round_time = time.time() - round_start
        
        # 打印轮次摘要
        if show_details:
            round_summary = self.server.get_round_summary(round_num)
            
            if client_accuracies:
                accs = list(client_accuracies.values())
                print(f"\n📊 Performance:")
                print(f"  Avg Accuracy: {np.mean(accs):.4f}")
                print(f"  Max Accuracy: {np.max(accs):.4f}")
                print(f"  Min Accuracy: {np.min(accs):.4f}")
            
            print(f"\n🎯 Contributions (CGSV):")
            contrib_stats = round_summary['contribution_stats']
            print(f"  Mean: {contrib_stats['mean']:.4f}, Std: {contrib_stats['std']:.4f}")
            print(f"  Range: [{contrib_stats['min']:.4f}, {contrib_stats['max']:.4f}]")
            
            if 'sparsification_stats' in round_summary and round_summary['sparsification_stats']:
                sparse_stats = round_summary['sparsification_stats']['by_level']
                print(f"\n✂️  Gradient Sparsification Statistics:")
                for level in ['diamond', 'gold', 'silver', 'bronze']:
                    if level in sparse_stats:
                        ls = sparse_stats[level]
                        print(f"  {level.capitalize()}: Keep={ls['avg_keep_ratio']:.3f}, "
                              f"Sparse={ls['avg_sparsity_rate']:.3f} (n={ls['count']})")
            
            print(f"\n⏱️  Time: {round_time:.2f}s")
            
            if round_num % 10 == 0 or round_num == self.args.num_rounds:
                self.membership_system.print_membership_distribution()
        
        # 记录指标
        round_metrics = {
            'round': round_num,
            'time_consumption': round_time,
            'num_selected_clients': len(selected_clients),
            'contributions': contributions.copy(),
            'client_accuracies': client_accuracies.copy(),
            'current_slice': current_slice,
            'active_points': all_active_points.copy(),
            'membership_levels': new_levels.copy(),
            'sparsification_stats': self.server.sparsification_distributor.get_sparsification_statistics()
        }
        
        self.metrics_calculator.record_round(round_metrics)
        return round_metrics
    
    def run_experiment(self):
        """运行完整实验"""
        print(f"\n{'='*80}")
        print(f"Starting Experiment: {self.experiment_name}")
        print(f"{'='*80}")
        
        # 独立训练基准
        self.compute_standalone_baselines()
        
        # 联邦学习训练
        print(f"\n{'='*80}")
        print("Federated Learning Training with Gradient Sparsification")
        print(f"{'='*80}")
        
        for round_num in range(1, self.args.num_rounds + 1):
            self.run_single_round(round_num)
        
        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        
        # 最终指标
        final_metrics = self.metrics_calculator.calculate_final_metrics()
        
        # 打印摘要
        self.metrics_calculator.print_summary()
        self.time_slice_manager.print_summary(self.args.num_rounds)
        self.server.print_contribution_summary()
        self.membership_system.print_membership_distribution()
        
        # 生成可视化
        self._generate_visualizations(final_metrics)
        
        # 保存结果
        self._save_results(final_metrics)
        
        return final_metrics
    
    def _generate_visualizations(self, final_metrics):
        """生成可视化"""
        print(f"\n{'='*80}")
        print("Generating Visualizations")
        print(f"{'='*80}")
        
        contributions_history = [rm.get('contributions', {}) 
                                for rm in self.metrics_calculator.round_metrics]
        
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
        """保存结果"""
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"{self.experiment_name}_results.json")
        
        save_data = {
            'experiment_name': self.experiment_name,
            'methodology': 'Gradient Sparsification-based Differentiated Distribution',
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
                'gradient_lr': self.args.gradient_lr,
                'standalone_epochs': self.args.standalone_epochs,
                'seed': self.args.seed
            },
            'sparsification_config': {
                'mode': self.args.sparsification_mode,
                'lambda': self.args.lambda_coef,
                'min_keep_ratio': self.args.min_keep_ratio,
                'sparsity_ranges': IncentiveConfig.LEVEL_SPARSITY_RANGES,
                'level_percentiles': IncentiveConfig.LEVEL_PERCENTILES
            },
            'final_metrics': final_metrics,
            'round_metrics': self.metrics_calculator.round_metrics[-10:],
            'membership_statistics': self.membership_system.get_membership_statistics(),
            'contribution_statistics': self.server.get_contribution_statistics(),
            'sparsification_statistics': self.server.sparsification_distributor.get_sparsification_statistics()
        }
        
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {results_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Federated Learning with Gradient Sparsification Incentive\n'
                    '联邦学习 - 基于梯度稀疏化的激励机制',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 使用示例:
  # 基础实验 - MNIST + IID
  python main.py --dataset mnist --distribution iid
  
  # Non-IID实验 - CIFAR10
  python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5
  
  # 调整稀疏化参数
  python main.py --dataset cifar10 --sparsification_mode magnitude --lambda_coef 2.0
  
  # 大规模实验
  python main.py --dataset cifar10 --num_clients 100 --num_rounds 100 \\
                 --sparsification_mode structured --lambda_coef 3.0
        """
    )
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100'],
                       help='Dataset name')
    
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients')
    
    # 数据分布
    parser.add_argument('--distribution', type=str, default='non-iid-dir',
                       choices=['iid', 'non-iid-dir'],
                       help='Data distribution type')
    
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-iid')
    
    # 训练参数
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of communication rounds')
    
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local epochs per round')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for local training')
    
    parser.add_argument('--gradient_lr', type=float, default=1.0,
                       help='Learning rate for applying sparse gradients (通常设为1.0)')
    
    parser.add_argument('--standalone_epochs', type=int, default=20,
                       help='Standalone training epochs')
    
    # 时间片参数
    parser.add_argument('--rounds_per_slice', type=int, default=5,
                       help='Rounds per time slice')
    
    # 稀疏化参数
    parser.add_argument('--sparsification_mode', type=str, default='magnitude',
                       choices=['magnitude', 'random', 'structured'],
                       help='Sparsification mode')
    
    parser.add_argument('--lambda_coef', type=float, default=2.0,
                       help='Lambda coefficient for keep ratio calculation')
    
    parser.add_argument('--min_keep_ratio', type=float, default=0.1,
                       help='Minimum keep ratio')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 运行实验
    experiment = GradientSparsificationFederatedLearning(args)
    final_metrics = experiment.run_experiment()
    
    # 打印最终结果
    print(f"\n{'='*80}")
    print("🎉 Experiment Completed!")
    print(f"{'='*80}")
    print(f"Experiment: {experiment.experiment_name}")
    print(f"\n📈 Key Results:")
    print(f"  Methodology: Gradient Sparsification")
    print(f"  Final Avg Accuracy: {final_metrics['client_accuracy']['avg_final']:.4f}")
    print(f"  PCC: {final_metrics['pcc']:.4f}")
    print(f"  IPR: {final_metrics['ipr']['final_ipr']:.4f} ({final_metrics['ipr']['ipr_percentage']:.2f}%)")
    print(f"  Total Time: {final_metrics['time_consumption']['total']:.2f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()