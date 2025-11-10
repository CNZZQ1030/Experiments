"""
main.py
主程序入口 / Main Program Entry
支持差异化模型奖励机制和动态客户端选择
Supports differentiated model reward mechanism and dynamic client selection
"""

import torch
import random
import numpy as np
import argparse
import json
import os
from datetime import datetime

# 导入配置 / Import configurations
from config import (
    FederatedConfig, IncentiveConfig, DatasetConfig, 
    ModelConfig, ExperimentConfig, DEVICE, SEED
)

# 导入实验模块 / Import experiment module
from experiments.run_experiments import ExperimentRunner


def set_seed(seed: int):
    """设置随机种子 / Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Federated Learning with Differentiated Model Rewards and Dynamic Selection'
    )
    
    # 基本参数 / Basic parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=100,
                       help='Number of training rounds')
    parser.add_argument('--clients_per_round', type=int, default=10,
                       help='Number of clients selected per round (used in fixed mode)')
    
    # 客户端选择策略参数 / Client selection strategy parameters
    parser.add_argument('--use_dynamic_selection', action='store_true',
                       help='Use dynamic client selection based on contribution threshold')
    parser.add_argument('--use_fixed_selection', action='store_true',
                       help='Use fixed number client selection (override config)')
    parser.add_argument('--min_participation_rate', type=float, default=None,
                       help='Minimum participation rate (for dynamic selection)')
    parser.add_argument('--max_participation_rate', type=float, default=None,
                       help='Maximum participation rate (for dynamic selection)')
    
    # 激励机制参数 / Incentive mechanism parameters
    parser.add_argument('--time_slice_type', type=str, default='rounds',
                       choices=['rounds', 'days', 'phases', 'dynamic', 'completion'],
                       help='Type of time slice strategy')
    parser.add_argument('--rounds_per_slice', type=int, default=10,
                       help='Rounds per time slice')
    
    # 差异化奖励参数 / Differentiated rewards parameters
    parser.add_argument('--enable_tiered_rewards', action='store_true',
                       help='Enable differentiated model rewards based on membership level')
    parser.add_argument('--tiered_strategy', type=str, default='weighted',
                       choices=['weighted', 'strict'],
                       help='Strategy for creating tiered models')
    
    # 数据分布参数 / Data distribution parameters
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non-iid'],
                       help='Data distribution type')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet distribution parameter for non-iid')
    
    # 实验参数 / Experiment parameters
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name of the experiment')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of experiment runs')
    parser.add_argument('--compare_methods', action='store_true',
                       help='Compare different time slice methods')
    parser.add_argument('--compare_rewards', action='store_true',
                       help='Compare standard vs differentiated rewards')
    parser.add_argument('--compare_selection', action='store_true',
                       help='Compare fixed vs dynamic selection strategies')
    
    # 其他参数 / Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    return parser.parse_args()


def main():
    """主函数 / Main function"""
    # 解析参数 / Parse arguments
    args = parse_arguments()
    
    # 设置随机种子 / Set random seed
    set_seed(args.seed)
    
    # 设置设备 / Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 更新配置 / Update configurations
    FederatedConfig.NUM_CLIENTS = args.num_clients
    FederatedConfig.NUM_ROUNDS = args.num_rounds
    FederatedConfig.CLIENTS_PER_ROUND = args.clients_per_round
    FederatedConfig.DATA_DISTRIBUTION = args.distribution
    FederatedConfig.NON_IID_ALPHA = args.alpha
    
    # 处理客户端选择策略配置
    if args.use_fixed_selection:
        FederatedConfig.USE_DYNAMIC_SELECTION = False
    elif args.use_dynamic_selection:
        FederatedConfig.USE_DYNAMIC_SELECTION = True
    # 否则使用配置文件中的默认值
    
    if args.min_participation_rate is not None:
        FederatedConfig.MIN_PARTICIPATION_RATE = args.min_participation_rate
    if args.max_participation_rate is not None:
        FederatedConfig.MAX_PARTICIPATION_RATE = args.max_participation_rate
    
    DatasetConfig.DATASET_NAME = args.dataset
    
    IncentiveConfig.TIME_SLICE_TYPE = args.time_slice_type
    IncentiveConfig.ROUNDS_PER_SLICE = args.rounds_per_slice
    IncentiveConfig.ENABLE_TIERED_REWARDS = args.enable_tiered_rewards
    IncentiveConfig.TIERED_AGGREGATION_STRATEGY = args.tiered_strategy
    
    # 设置实验名称 / Set experiment name
    if args.experiment_name is None:
        reward_type = "tiered" if args.enable_tiered_rewards else "standard"
        selection_type = "dynamic" if FederatedConfig.USE_DYNAMIC_SELECTION else "fixed"
        args.experiment_name = (f"FL_{args.dataset}_{reward_type}_{selection_type}_"
                               f"{args.time_slice_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    print(f"\nStarting experiment: {args.experiment_name}")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Distribution: {args.distribution}")
    print(f"  Time Slice Type: {args.time_slice_type}")
    print(f"  Client Selection: {'Dynamic' if FederatedConfig.USE_DYNAMIC_SELECTION else 'Fixed'}")
    print(f"  Differentiated Rewards: {'Enabled' if args.enable_tiered_rewards else 'Disabled'}")
    if args.enable_tiered_rewards:
        print(f"  Tiered Strategy: {args.tiered_strategy}")
    
    if args.compare_selection:
        # 对比固定选择 vs 动态选择 / Compare fixed vs dynamic selection
        print("\n" + "="*70)
        print("COMPARING FIXED VS DYNAMIC CLIENT SELECTION")
        print("="*70)
        
        results_comparison = {}
        
        for use_dynamic in [False, True]:
            selection_name = "Dynamic" if use_dynamic else "Fixed"
            
            print(f"\n{'='*70}")
            print(f"Running experiment with {selection_name} Selection")
            print('='*70)
            
            # 创建实验运行器
            runner = ExperimentRunner(
                dataset_name=args.dataset,
                num_clients=args.num_clients,
                num_rounds=args.num_rounds,
                clients_per_round=args.clients_per_round,
                time_slice_type=args.time_slice_type,
                distribution=args.distribution,
                device=device,
                use_dynamic_selection=use_dynamic
            )
            
            # 运行实验
            exp_name = f"{args.experiment_name}_{selection_name.lower()}"
            results = runner.run_single_experiment(
                experiment_name=exp_name,
                num_runs=1
            )
            
            results_comparison[selection_name] = results
        
        # 保存对比结果
        comparison_path = os.path.join(
            ExperimentConfig.LOG_DIR, 
            f"comparison_selection_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(comparison_path, 'w') as f:
            json.dump(results_comparison, f, indent=2)
        
        # 打印对比总结
        print("\n" + "="*70)
        print("SELECTION STRATEGY COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Metric':<35} {'Fixed':<15} {'Dynamic':<15} {'Improvement':<15}")
        print("-"*70)
        
        metrics_to_compare = [
            ('Final Accuracy', 'accuracy_final'),
            ('Average Accuracy', 'accuracy_avg'),
            ('Final Loss', 'loss_final'),
            ('Avg Participation Rate', 'participation_rate_avg'),
            ('Participation Rate Std', 'participation_rate_std'),
            ('System Activity', 'system_activity_avg'),
            ('IPR (Accuracy)', 'ipr_accuracy_final'),
        ]
        
        for metric_name, metric_key in metrics_to_compare:
            fixed_val = results_comparison['Fixed'].get(metric_key, 0)
            dynamic_val = results_comparison['Dynamic'].get(metric_key, 0)
            
            if metric_key in ['loss_final', 'participation_rate_std']:
                # 对于损失和标准差，越低越好
                improvement = ((fixed_val - dynamic_val) / fixed_val * 100) if fixed_val != 0 else 0
            else:
                # 对于准确率等指标，越高越好
                improvement = ((dynamic_val - fixed_val) / fixed_val * 100) if fixed_val != 0 else 0
            
            print(f"{metric_name:<35} {fixed_val:<15.4f} {dynamic_val:<15.4f} {improvement:>+14.2f}%")
        
        print("="*70)
        print(f"Comparison results saved to {comparison_path}")
    
    elif args.compare_rewards:
        # 对比标准激励 vs 差异化奖励
        print("\n" + "="*70)
        print("COMPARING STANDARD VS DIFFERENTIATED REWARDS")
        print("="*70)
        
        results_comparison = {}
        
        for enable_tiered in [False, True]:
            IncentiveConfig.ENABLE_TIERED_REWARDS = enable_tiered
            reward_name = "Differentiated" if enable_tiered else "Standard"
            
            print(f"\n{'='*70}")
            print(f"Running experiment with {reward_name} Rewards")
            print('='*70)
            
            runner = ExperimentRunner(
                dataset_name=args.dataset,
                num_clients=args.num_clients,
                num_rounds=args.num_rounds,
                clients_per_round=args.clients_per_round,
                time_slice_type=args.time_slice_type,
                distribution=args.distribution,
                device=device
            )
            
            exp_name = f"{args.experiment_name}_{reward_name.lower()}"
            results = runner.run_single_experiment(
                experiment_name=exp_name,
                num_runs=1
            )
            
            results_comparison[reward_name] = results
        
        # 保存对比结果
        comparison_path = os.path.join(
            ExperimentConfig.LOG_DIR, 
            f"comparison_rewards_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(comparison_path, 'w') as f:
            json.dump(results_comparison, f, indent=2)
        
        # 打印对比总结
        print("\n" + "="*70)
        print("REWARDS COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Metric':<30} {'Standard':<15} {'Differentiated':<15} {'Improvement':<15}")
        print("-"*70)
        
        metrics_to_compare = [
            ('Final Accuracy', 'accuracy_final'),
            ('Average Accuracy', 'accuracy_avg'),
            ('Final Loss', 'loss_final'),
            ('Participation Rate', 'participation_rate_avg'),
            ('System Activity', 'system_activity_avg'),
            ('Convergence Round', 'convergence_round')
        ]
        
        for metric_name, metric_key in metrics_to_compare:
            standard_val = results_comparison['Standard'].get(metric_key, 0)
            tiered_val = results_comparison['Differentiated'].get(metric_key, 0)
            
            if metric_key in ['loss_final', 'convergence_round']:
                improvement = ((standard_val - tiered_val) / standard_val * 100) if standard_val != 0 else 0
            else:
                improvement = ((tiered_val - standard_val) / standard_val * 100) if standard_val != 0 else 0
            
            print(f"{metric_name:<30} {standard_val:<15.4f} {tiered_val:<15.4f} {improvement:>+14.2f}%")
        
        if 'quality_gap_avg' in results_comparison['Differentiated']:
            print(f"\nModel Quality Gap (Diamond-Bronze): {results_comparison['Differentiated']['quality_gap_avg']:.4f}")
        
        print("="*70)
        print(f"Comparison results saved to {comparison_path}")
        
    elif args.compare_methods:
        # 比较不同的时间片方法
        print("\nComparing different time slice methods...")
        
        runner = ExperimentRunner(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            clients_per_round=args.clients_per_round,
            time_slice_type=args.time_slice_type,
            distribution=args.distribution,
            device=device
        )
        
        results = runner.compare_time_slice_methods()
        
        # 保存比较结果
        results_path = os.path.join(
            ExperimentConfig.LOG_DIR,
            f"comparison_methods_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Comparison results saved to {results_path}")
        
    else:
        # 运行单个实验
        print(f"\nRunning single experiment...")
        
        runner = ExperimentRunner(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            clients_per_round=args.clients_per_round,
            time_slice_type=args.time_slice_type,
            distribution=args.distribution,
            device=device
        )
        
        results = runner.run_single_experiment(
            experiment_name=args.experiment_name,
            num_runs=args.num_runs
        )
        
        # 打印结果摘要
        results_path = os.path.join(ExperimentConfig.LOG_DIR, f"{args.experiment_name}_results.json")
        
        print("\n" + "="*70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*70)
        print(f"Final Accuracy: {results['accuracy_final']:.4f}")
        print(f"Final Loss: {results['loss_final']:.4f}")
        print(f"Average Participation Rate: {results['participation_rate_avg']:.4f}")
        
        if 'participation_rate_std' in results:
            print(f"Participation Rate Std: {results['participation_rate_std']:.4f}")
        
        if 'qualified_rate_avg' in results:
            print(f"Average Qualified Rate: {results['qualified_rate_avg']:.4f}")
            print(f"Average Selection Efficiency: {results.get('selection_efficiency_avg', 0):.4f}")
        
        print(f"Average System Activity: {results['system_activity_avg']:.4f}")
        print(f"Convergence Round: {results.get('convergence_round', 'N/A')}")
        
        if args.enable_tiered_rewards and 'quality_gap_avg' in results:
            print(f"\nDifferentiated Rewards Metrics:")
            print(f"  Average Quality Gap (Diamond-Bronze): {results['quality_gap_avg']:.4f}")
            print(f"  Final Quality Gap: {results.get('quality_gap_final', 0):.4f}")
        
        if FederatedConfig.USE_DYNAMIC_SELECTION and 'selection_mode_distribution' in results:
            print(f"\nSelection Mode Distribution:")
            for mode, count in results['selection_mode_distribution'].items():
                print(f"  {mode}: {count} rounds")
        
        print("="*70)
        print(f"\nAll results saved in:")
        print(f"  Results JSON: {results_path}")
        print(f"  Plots: {ExperimentConfig.PLOTS_DIR}")
        print(f"  Checkpoints: {os.path.join(ExperimentConfig.CHECKPOINT_DIR, args.experiment_name)}/")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(ExperimentConfig.BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ExperimentConfig.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ExperimentConfig.LOG_DIR, exist_ok=True)
    os.makedirs(ExperimentConfig.PLOTS_DIR, exist_ok=True)
    
    main()