"""
main.py - 层级约束动态梯度奖励实验主程序（重构版）
Main Program - Tier-Constrained Dynamic Gradient Reward Experiment (Refactored)

基于NeurIPS 2021论文"Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"
Based on NeurIPS 2021 paper "Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning"

核心创新 / Core Innovations:
1. 层级作为稀疏率的上下界（Bounds）/ Tiers as bounds for keep ratios
2. 组内插值（Intra-Tier Interpolation）/ Intra-tier interpolation
3. 大幅降低低贡献客户端的参数保留率以提高PCC / Significantly reduce keep ratio for low-contribution clients

修复说明 / Bug Fix:
- 修复了梯度计算基准点的问题
- 在步骤6中，客户端应用稀疏梯度后，立即更新服务器记录的client_previous_weights
- 确保下一轮梯度计算：Δw_i = w_i^new - w_local_i^(应用稀疏梯度后)

使用方法 / Usage:
    # 基础实验 - CIFAR10 + Non-IID
    python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5
    
    # 使用激进配置（更大差异化）
    python main.py --dataset cifar10 --tier_config aggressive
    
    # 使用温和配置
    python main.py --dataset cifar10 --tier_config moderate
    
    # 完整参数示例
    python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5 \\
                   --num_clients 100 --num_rounds 100 --tier_config default \\
                   --sparsification_mode magnitude --aggregation_method contribution
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

# 添加项目路径 / Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FederatedConfig, IncentiveConfig, DatasetConfig, DEVICE
from datasets.data_loader import FederatedDataLoader
from models.cnn_model import ModelFactory
from incentive.time_slice import TimeSliceManager
from incentive.membership import MembershipSystem
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer

# 导入重构的服务器和客户端 / Import refactored server and client
from federated.server import FederatedServerWithGradientSparsification
from federated.client import FederatedClient


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


class TierConstrainedFederatedLearning:
    """
    层级约束动态梯度奖励联邦学习实验
    Tier-Constrained Dynamic Gradient Reward Federated Learning Experiment
    
    核心创新 / Core Innovations:
    1. 聚合全局梯度而非模型权重 / Aggregate global gradients instead of model weights
    2. 使用层级约束进行差异化稀疏 / Use tier constraints for differential sparsification
    3. 组内插值实现连续映射 / Intra-tier interpolation for continuous mapping
    4. 基于幅度的稀疏化保留最重要参数 / Magnitude-based pruning to retain important parameters
    
    正确的训练流程 / Correct Training Workflow:
    1. 客户端本地训练：w_i^t = LocalTrain(w_local_i^(t-1))
    2. 服务器计算梯度：Δw_i = w_i^t - w_local_i^(t-1)
    3. 聚合全局梯度：Δw_global = Aggregate(Δw_i)
    4. 稀疏化分发：sparse_i(Δw_global)
    5. 客户端应用：w_local_i^t = w_i^t + lr * sparse_i(Δw_global)
    6. 更新基准点：client_previous_weights[i] = w_local_i^t
    """
    
    def __init__(self, args):
        """初始化实验 / Initialize experiment"""
        self.args = args
        set_seed(args.seed)
        
        self.device = DEVICE
        self.experiment_name = self._generate_experiment_name()
        
        print(f"\n{'='*80}")
        print(f"Tier-Constrained Dynamic Gradient Reward Federated Learning")
        print(f"层级约束动态梯度奖励联邦学习")
        print(f"{'='*80}")
        print(f"Experiment / 实验名称: {self.experiment_name}")
        print(f"Dataset / 数据集: {args.dataset}")
        print(f"Distribution / 分布: {args.distribution}")
        if args.distribution == "non-iid-dir":
            print(f"  Alpha: {args.alpha}")
        print(f"Clients / 客户端数: {args.num_clients}")
        print(f"Rounds / 轮次: {args.num_rounds}")
        print(f"Device / 设备: {self.device}")
        
        print(f"\n✨ Tier-Constrained Configuration / 层级约束配置:")
        print(f"  Tier Config / 层级配置: {args.tier_config}")
        print(f"  Sparsification Mode / 稀疏化模式: {args.sparsification_mode}")
        print(f"  Aggregation Method / 聚合方式: {args.aggregation_method}")
        print(f"  Gradient Application LR / 梯度应用学习率: {args.gradient_lr}")
        
        # 显示层级保留率范围 / Show tier keep ratio ranges
        if args.tier_config == "aggressive":
            tier_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES_AGGRESSIVE
        elif args.tier_config == "moderate":
            tier_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES_MODERATE
        else:
            tier_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES
        
        print(f"\n📊 Tier Keep Ratio Ranges / 层级保留率范围:")
        for tier, (low, high) in tier_ranges.items():
            sparsity_low, sparsity_high = 1.0 - high, 1.0 - low
            print(f"    {tier.capitalize():8s}: Keep [{low:.2f}, {high:.2f}] "
                  f"(Sparsity [{sparsity_low:.2f}, {sparsity_high:.2f}])")
        print(f"{'='*80}")
        
        # 更新配置 / Update configuration
        IncentiveConfig.SPARSIFICATION_MODE = args.sparsification_mode
        
        self._initialize_components()
    
    def _generate_experiment_name(self) -> str:
        """生成实验名称 / Generate experiment name"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dist_suffix = f"_a{self.args.alpha}" if self.args.distribution == "non-iid-dir" else ""
        tier_suffix = f"_TierConstrained_{self.args.tier_config}_{self.args.sparsification_mode}"
        return f"{self.args.dataset}_{self.args.distribution}{dist_suffix}" \
               f"_c{self.args.num_clients}_r{self.args.num_rounds}{tier_suffix}_{timestamp}"
    
    def _initialize_components(self):
        """初始化所有组件 / Initialize all components"""
        print("\nInitializing components / 初始化组件...")
        
        # 1. 数据加载 / Data loading
        print("  [1/6] Loading data / 加载数据...")
        self.data_loader = FederatedDataLoader(
            dataset_name=self.args.dataset,
            num_clients=self.args.num_clients,
            batch_size=self.args.batch_size,
            distribution=self.args.distribution,
            alpha=self.args.alpha
        )
        
        # 2. 模型创建 / Model creation
        print("  [2/6] Creating model / 创建模型...")
        num_classes = DatasetConfig.NUM_CLASSES[self.args.dataset]
    
        # 判断是否为文本数据集 / Check if it's a text dataset
        if DatasetConfig.is_text_dataset(self.args.dataset):
            # ===== 文本数据集处理 / Text dataset handling =====
            vocab_size = self.data_loader.get_vocab_size()
            text_config = DatasetConfig.get_text_config(self.args.dataset)
        
            # 创建文本模型 / Create text model
            self.model = ModelFactory.create_model(
                self.args.dataset,
                num_classes=num_classes,
                vocab_size=vocab_size,
                **text_config
            )
            print(f"     Text model created with vocab_size={vocab_size}")
        else:
            # ===== 图像数据集处理 / Image dataset handling =====
            input_channels = DatasetConfig.INPUT_SHAPE[self.args.dataset][0]
        
            # 创建图像模型 / Create image model
            self.model = ModelFactory.create_model(
                self.args.dataset,
                num_classes=num_classes,
                input_channels=input_channels
            )
    
        # 打印模型信息 / Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f" Model parameters / 模型参数数: {total_params:,}")
        
        # 3. 服务器初始化（层级约束版本）/ Server initialization (tier-constrained version)
        print("  [3/6] Initializing server with Tier-Constrained Gradient Sparsification...")
        print("  初始化层级约束梯度稀疏化服务器...")
        self.server = FederatedServerWithGradientSparsification(
            model=self.model, 
            device=self.device,
            tier_config=self.args.tier_config,
            aggregation_method=self.args.aggregation_method
        )
        
        # 4. 客户端创建 / Client creation
        print("  [4/6] Creating clients / 创建客户端...")
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
        print("  [5/6] Initializing incentive system / 初始化激励系统...")
        self.time_slice_manager = TimeSliceManager(
            slice_type="rounds",
            rounds_per_slice=self.args.rounds_per_slice,
            validity_slices=IncentiveConfig.POINTS_VALIDITY_SLICES
        )
        
        # 使用三级制会员系统 / Use three-tier membership system
        self.membership_system = MembershipSystem(
            ranking_percentiles=IncentiveConfig.LEVEL_PERCENTILES,
            use_three_tier=True
        )
        
        for client_id in range(self.args.num_clients):
            self.membership_system.initialize_client(client_id)
        
        # 6. 指标系统 / Metrics system
        print("  [6/6] Initializing metrics / 初始化指标...")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(output_dir="outputs/figures")
        
        print("✓ All components initialized / 所有组件初始化完成")
    
    def compute_standalone_baselines(self):
        """计算独立训练基准 / Compute standalone baselines"""
        print(f"\n{'='*80}")
        print(f"Computing Standalone Baselines ({self.args.standalone_epochs} epochs)")
        print(f"计算独立训练基准")
        print(f"{'='*80}")
        
        for client_id, client in tqdm(self.clients.items(), desc="Standalone training"):
            standalone_acc, _ = client.train_standalone(epochs=self.args.standalone_epochs)
            self.metrics_calculator.record_standalone_accuracy(client_id, standalone_acc)
        
        # 打印基准统计 / Print baseline statistics
        standalone_accs = list(self.metrics_calculator.standalone_accuracies.values())
        print(f"\nStandalone Baseline Statistics / 独立训练基准统计:")
        print(f"  Mean / 均值: {np.mean(standalone_accs):.4f}")
        print(f"  Std / 标准差: {np.std(standalone_accs):.4f}")
        print(f"  Range / 范围: [{np.min(standalone_accs):.4f}, {np.max(standalone_accs):.4f}]")
        print("✓ Standalone baselines computed / 独立训练基准计算完成")
    
    def run_single_round(self, round_num: int) -> dict:
        """
        运行单轮训练 - 层级约束动态梯度奖励版本
        Run single round - Tier-Constrained Dynamic Gradient Reward version
        
        核心流程 / Core Workflow:
        1. 客户端本地训练，上传训练后的权重
        2. 服务器计算客户端梯度：Δw_i = w_i^new - w_i^old
        3. 聚合全局梯度（可选贡献度加权）
        4. 计算CGSV贡献度和更新会员等级
        5. 使用层级约束进行差异化稀疏（组内插值）
        6. 客户端应用稀疏梯度，并更新服务器记录的基准点
        """
        round_start = time.time()
        
        selected_clients = list(range(self.args.num_clients))
        self.server.reset_round()
        client_accuracies = {}
        
        # 判断是否显示详细信息 / Determine whether to show details
        show_details = (round_num % max(1, self.args.num_rounds // 10) == 0) or \
                      round_num == 1 or round_num == self.args.num_rounds
        
        if show_details:
            print(f"\n{'='*80}")
            print(f"Round {round_num}/{self.args.num_rounds}")
            print(f"{'='*80}")
        
        # ========== 步骤1: 客户端本地训练 ==========
        # ========== Step 1: Client local training ==========
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num} - Training",
                            leave=False):
            client = self.clients[client_id]
            
            # 第一轮需要初始化本地模型 / First round needs to initialize local model
            if round_num == 1:
                global_weights = self.server.get_global_model_weights()
            else:
                global_weights = None  # 后续轮次使用本地模型
            
            # 本地训练 / Local training
            updated_weights, train_info = client.train_federated(
                global_weights=global_weights,
                epochs=self.args.local_epochs,
                lr=self.args.learning_rate
            )
            
            # 服务器收集更新 / Server collects updates
            self.server.collect_client_updates(client_id, updated_weights, train_info)
            
            # 记录准确率 / Record accuracy
            federated_acc = train_info['federated_accuracy']
            self.metrics_calculator.record_federated_accuracy(client_id, federated_acc)
            client_accuracies[client_id] = federated_acc
        
        # ========== 步骤2: 聚合全局梯度 ==========
        # ========== Step 2: Aggregate global gradient ==========
        self.server.update_global_model()
        
        # ========== 步骤3: 计算CGSV贡献度 ==========
        # ========== Step 3: Calculate CGSV contributions ==========
        contributions = self.server.calculate_all_contributions(round_num)
        
        # ========== 步骤4: 时间片积分和会员等级 ==========
        # ========== Step 4: Time slice points and membership levels ==========
        all_active_points = {}
        for client_id, contribution in contributions.items():
            active_points = self.time_slice_manager.add_contribution_points(
                client_id, round_num, contribution
            )
            all_active_points[client_id] = active_points
            
            # 更新贡献历史 / Update contribution history
            self.membership_system.update_contribution_history(client_id, contribution, round_num)
        
        # 更新会员等级 / Update membership levels
        new_levels = self.membership_system.update_all_memberships_by_ranking(all_active_points)
        for client_id, new_level in new_levels.items():
            self.clients[client_id].update_membership_level(new_level)
        
        # 清理过期积分 / Clean expired points
        current_slice = self.time_slice_manager.get_current_slice(round_num)
        if round_num > 1:
            prev_slice = self.time_slice_manager.get_current_slice(round_num - 1)
            if current_slice != prev_slice:
                cleaned = self.time_slice_manager.clean_expired_points(round_num)
                if cleaned and show_details:
                    print(f"  Time slice changed / 时间片变化: {prev_slice} → {current_slice}")
                
                # 重新计算等级 / Recalculate levels
                updated_points = self.time_slice_manager.get_all_client_active_points(round_num)
                new_levels = self.membership_system.update_all_memberships_by_ranking(updated_points)
                for client_id, new_level in new_levels.items():
                    self.clients[client_id].update_membership_level(new_level)
        
        # ========== 步骤5: 分发层级约束稀疏化梯度 ==========
        # ========== Step 5: Distribute tier-constrained sparsified gradients ==========
        sparsified_gradients = self.server.distribute_sparsified_gradients(new_levels)
        
        # ========== 步骤6: 客户端应用稀疏梯度 ==========
        # ========== Step 6: Clients apply sparse gradients ==========
        for client_id in tqdm(selected_clients, 
                            desc=f"Round {round_num} - Applying gradients",
                            leave=False):
            if client_id in sparsified_gradients:
                sparse_gradient = sparsified_gradients[client_id]
                self.clients[client_id].apply_gradient_update(
                    sparse_gradient, 
                    learning_rate=self.args.gradient_lr
                )
                
                # 关键：更新服务器端记录的客户端"上一轮"权重
                # CRITICAL: Update server's record of client's "previous round" weights
                # 这确保下一轮梯度计算的基准点是正确的（应用稀疏梯度后的状态）
                # This ensures the gradient baseline for next round is correct
                # (state after applying sparse gradient)
                current_weights = self.clients[client_id].get_local_model_weights()
                self.server.update_client_previous_weights(client_id, current_weights)
        
        round_time = time.time() - round_start
        
        # 打印轮次摘要 / Print round summary
        if show_details:
            round_summary = self.server.get_round_summary(round_num)
            
            if client_accuracies:
                accs = list(client_accuracies.values())
                print(f"\n📊 Performance / 性能:")
                print(f"  Avg Accuracy / 平均准确率: {np.mean(accs):.4f}")
                print(f"  Max Accuracy / 最大准确率: {np.max(accs):.4f}")
                print(f"  Min Accuracy / 最小准确率: {np.min(accs):.4f}")
            
            print(f"\n🎯 Contributions (CGSV) / 贡献度:")
            contrib_stats = round_summary['contribution_stats']
            print(f"  Mean / 均值: {contrib_stats['mean']:.4f}, Std / 标准差: {contrib_stats['std']:.4f}")
            print(f"  Range / 范围: [{contrib_stats['min']:.4f}, {contrib_stats['max']:.4f}]")
            
            # 打印稀疏化统计 / Print sparsification statistics
            if 'sparsification_stats' in round_summary and round_summary['sparsification_stats']:
                sparse_stats = round_summary['sparsification_stats']
                if 'by_level' in sparse_stats:
                    print(f"\n✂️  Tier-Constrained Sparsification / 层级约束稀疏化:")
                    for level in ['gold', 'silver', 'bronze']:
                        if level in sparse_stats['by_level']:
                            ls = sparse_stats['by_level'][level]
                            print(f"  {level.capitalize():8s}: Keep={ls['mean']:.3f}, "
                                  f"Range=[{ls['min']:.3f}, {ls['max']:.3f}] (n={ls['count']})")
            
            print(f"\n⏱️  Time / 耗时: {round_time:.2f}s")
            
            # 定期打印会员分布 / Periodically print membership distribution
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
            'membership_levels': new_levels.copy(),
            'sparsification_stats': self.server.sparsification_distributor.get_sparsification_statistics()
        }
        
        self.metrics_calculator.record_round(round_metrics)
        return round_metrics
    
    def run_experiment(self):
        """运行完整实验 / Run complete experiment"""
        print(f"\n{'='*80}")
        print(f"Starting Experiment / 开始实验: {self.experiment_name}")
        print(f"{'='*80}")
        
        # 独立训练基准 / Standalone baselines
        self.compute_standalone_baselines()
        
        # 联邦学习训练 / Federated learning training
        print(f"\n{'='*80}")
        print("Federated Learning with Tier-Constrained Dynamic Gradient Reward")
        print("层级约束动态梯度奖励联邦学习")
        print(f"{'='*80}")
        
        for round_num in range(1, self.args.num_rounds + 1):
            self.run_single_round(round_num)
        
        print(f"\n{'='*80}")
        print("Training Complete / 训练完成")
        print(f"{'='*80}")
        
        # 最终指标 / Final metrics
        final_metrics = self.metrics_calculator.calculate_final_metrics()
        
        # 打印摘要 / Print summaries
        self.metrics_calculator.print_summary()
        self.time_slice_manager.print_summary(self.args.num_rounds)
        self.server.print_contribution_summary()
        self.server.print_sparsification_summary()
        self.membership_system.print_membership_distribution()
        
        # 生成可视化 / Generate visualizations
        self._generate_visualizations(final_metrics)
        
        # 保存结果 / Save results
        self._save_results(final_metrics)
        
        return final_metrics
    
    def _generate_visualizations(self, final_metrics):
        """生成可视化 / Generate visualizations"""
        print(f"\n{'='*80}")
        print("Generating Visualizations / 生成可视化")
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
        print("✓ Visualizations generated / 可视化生成完成")
    
    def _save_results(self, final_metrics):
        """保存结果 / Save results"""
        results_dir = "outputs/results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"{self.experiment_name}_results.json")
        
        # 获取层级配置 / Get tier configuration
        if self.args.tier_config == "aggressive":
            tier_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES_AGGRESSIVE
        elif self.args.tier_config == "moderate":
            tier_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES_MODERATE
        else:
            tier_ranges = IncentiveConfig.TIER_KEEP_RATIO_RANGES
        
        save_data = {
            'experiment_name': self.experiment_name,
            'methodology': 'Tier-Constrained Dynamic Gradient Reward',
            'methodology_cn': '层级约束动态梯度奖励',
            'reference': 'NeurIPS 2021 - Gradient-Driven Rewards to Guarantee Fairness in Collaborative Machine Learning',
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
            'tier_constrained_config': {
                'tier_config': self.args.tier_config,
                'sparsification_mode': self.args.sparsification_mode,
                'aggregation_method': self.args.aggregation_method,
                'tier_keep_ratio_ranges': {k: list(v) for k, v in tier_ranges.items()},
                'level_percentiles': IncentiveConfig.LEVEL_PERCENTILES
            },
            'final_metrics': final_metrics,
            'round_metrics_last_10': self.metrics_calculator.round_metrics[-10:],
            'membership_statistics': self.membership_system.get_membership_statistics(),
            'contribution_statistics': self.server.get_contribution_statistics(),
            'sparsification_statistics': self.server.sparsification_distributor.get_sparsification_statistics()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✓ Results saved to / 结果已保存至: {results_path}")


def parse_args():
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Tier-Constrained Dynamic Gradient Reward Federated Learning\n'
                    '层级约束动态梯度奖励联邦学习',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 使用示例:
  # 基础实验 - MNIST + IID
  python main.py --dataset mnist --distribution iid
  
  # Non-IID实验 - CIFAR10（默认配置）
  python main.py --dataset cifar10 --distribution non-iid-dir --alpha 0.5
  
  # 使用激进配置（更大差异化，适合提高PCC）
  python main.py --dataset cifar10 --tier_config aggressive
  
  # 使用温和配置（更均衡的差异化）
  python main.py --dataset cifar10 --tier_config moderate
  
  # 大规模实验
  python main.py --dataset cifar10 --num_clients 100 --num_rounds 100 \\
                 --tier_config default --sparsification_mode magnitude
                 
  # 对比实验 - 使用FedAvg聚合
  python main.py --dataset cifar10 --aggregation_method fedavg
        """
    )
    
    # 数据集参数 / Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'mr', 'sst'],
                       help='Dataset name / 数据集名称\n'
                            '  Image: mnist, fashion-mnist, cifar10, cifar100\n'
                            '  Text: mr (Movie Review), sst (Stanford Sentiment Treebank)')
    
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients / 客户端数量')
    
    # 数据分布 / Data distribution
    parser.add_argument('--distribution', type=str, default='non-iid-dir',
                       choices=['iid', 'non-iid-dir'],
                       help='Data distribution type / 数据分布类型')
    
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-iid / Non-IID的Dirichlet参数')
    
    # 训练参数 / Training parameters
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of communication rounds / 通信轮次')
    
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local epochs per round / 每轮本地训练轮次')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size / 批次大小')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for local training / 本地训练学习率')
    
    parser.add_argument('--gradient_lr', type=float, default=1.0,
                       help='Learning rate for applying sparse gradients / 稀疏梯度应用学习率（其实不该叫学习率，就是一个权重系数）')
    
    parser.add_argument('--standalone_epochs', type=int, default=20,
                       help='Standalone training epochs / 独立训练轮次')
    
    # 时间片参数 / Time slice parameters
    parser.add_argument('--rounds_per_slice', type=int, default=5,
                       help='Rounds per time slice / 每个时间片的轮次')
    
    # 层级约束参数 / Tier-constrained parameters
    parser.add_argument('--tier_config', type=str, default='default',
                       choices=['default', 'aggressive', 'moderate'],
                       help='Tier configuration / 层级配置\n'
                            '  default: Gold[0.8,1.0], Silver[0.5,0.8], Bronze[0.1,0.5]\n'
                            '  aggressive: 更大差异化 / More differentiation\n'
                            '  moderate: 更温和 / More moderate')
    
    parser.add_argument('--sparsification_mode', type=str, default='magnitude',
                       choices=['magnitude', 'random', 'structured'],
                       help='Sparsification mode / 稀疏化模式\n'
                            '  magnitude: 基于幅度（推荐）\n'
                            '  random: 随机\n'
                            '  structured: 结构化')
    
    parser.add_argument('--aggregation_method', type=str, default='contribution',
                       choices=['fedavg', 'contribution'],
                       help='Aggregation method / 聚合方式\n'
                            '  fedavg: 基于样本数量\n'
                            '  contribution: 基于贡献度')
    
    # 其他参数 / Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed / 随机种子')
    
    return parser.parse_args()


def main():
    """主函数 / Main function"""
    args = parse_args()
    
    # 运行实验 / Run experiment
    experiment = TierConstrainedFederatedLearning(args)
    final_metrics = experiment.run_experiment()
    
    # 打印最终结果 / Print final results
    print(f"\n{'='*80}")
    print("🎉 Experiment Completed! / 实验完成！")
    print(f"{'='*80}")
    print(f"Experiment / 实验: {experiment.experiment_name}")
    print(f"\n📈 Key Results / 关键结果:")
    print(f"  Methodology / 方法: Tier-Constrained Dynamic Gradient Reward")
    print(f"  Final Avg Accuracy / 最终平均准确率: {final_metrics['client_accuracy']['avg_final']:.4f}")
    print(f"  PCC: {final_metrics['pcc']:.4f}")
    print(f"  IPR: {final_metrics['ipr']['final_ipr']:.4f} ({final_metrics['ipr']['ipr_percentage']:.2f}%)")
    print(f"  Total Time / 总耗时: {final_metrics['time_consumption']['total']:.2f}s")
    
    # PCC结果解读 / PCC result interpretation
    pcc = final_metrics['pcc']
    print(f"\n📊 PCC Interpretation / PCC解读:")
    if pcc >= 0.75:
        print(f"  ✓ Excellent! Strong positive correlation / 极好！强正相关")
        print(f"    激励机制效果显著，高贡献客户端获得更好性能")
    elif pcc >= 0.65:
        print(f"  ✓ Good! Moderate positive correlation / 良好！中等正相关")
        print(f"    激励机制有效，贡献与收益呈正相关")
    elif pcc >= 0.5:
        print(f"  △ Fair. Weak positive correlation / 一般。弱正相关")
        print(f"    激励机制有一定效果，可考虑使用aggressive配置")
    else:
        print(f"  △ Need improvement / 需要改进")
        print(f"    建议：python main.py --tier_config aggressive")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()