"""
main.py
主程序入口 / Main Entry Point
支持AMAC贡献度计算和差异化模型分发
Supports AMAC contribution calculation and differentiated model distribution
"""

import torch
import argparse
import random
import numpy as np
import os
import sys
from datetime import datetime

# 添加路径 / Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.experiment_runner import ExperimentRunner, compare_experiments
from config import SEED, DEVICE


def set_seed(seed: int):
    """设置随机种子 / Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments():
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Federated Learning with AMAC and Differentiated Model Distribution / '
                    '基于AMAC和差异化模型分发的联邦学习'
    )
    
    # 基本参数 / Basic parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use / 使用的数据集')
    
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Number of clients / 客户端数量')
    
    parser.add_argument('--num_rounds', type=int, default=100,
                       help='Number of training rounds / 训练轮次')
    
    # 数据分布 / Data distribution
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non-iid'],
                       help='Data distribution type / 数据分布类型')
    
    # 时间片策略 / Time slice strategy
    parser.add_argument('--time_slice', type=str, default='rounds',
                       choices=['rounds', 'days', 'phases', 'dynamic', 'completion'],
                       help='Time slice type / 时间片类型')
    
    # 实验设置 / Experiment settings
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name of the experiment / 实验名称')
    
    parser.add_argument('--compare_distributions', action='store_true',
                       help='Compare IID vs Non-IID / 比较IID和Non-IID')
    
    parser.add_argument('--compare_time_slices', action='store_true',
                       help='Compare different time slice methods / 比较不同时间片方法')
    
    parser.add_argument('--compare_datasets', action='store_true',
                       help='Compare different datasets / 比较不同数据集')
    
    # 其他参数 / Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed / 随机种子')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use / 使用的设备')
    
    return parser.parse_args()


def main():
    """主函数 / Main function"""
    args = parse_arguments()
    
    # 设置随机种子 / Set random seed
    set_seed(args.seed)
    
    # 设置设备 / Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 生成实验名称 / Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"AMAC_{args.dataset}_{args.distribution}_{args.time_slice}_{timestamp}"
    
    # 运行比较实验 / Run comparison experiments
    if args.compare_distributions:
        # 比较IID vs Non-IID / Compare IID vs Non-IID
        print("\n" + "="*70)
        print("Comparing IID vs Non-IID Distributions / 比较IID与Non-IID分布")
        print("="*70)
        
        configurations = [
            {
                'name': f'{args.dataset}_iid_{args.time_slice}',
                'dataset': args.dataset,
                'num_clients': args.num_clients,
                'num_rounds': args.num_rounds,
                'distribution': 'iid',
                'time_slice_type': args.time_slice,
                'device': device
            },
            {
                'name': f'{args.dataset}_non-iid_{args.time_slice}',
                'dataset': args.dataset,
                'num_clients': args.num_clients,
                'num_rounds': args.num_rounds,
                'distribution': 'non-iid',
                'time_slice_type': args.time_slice,
                'device': device
            }
        ]
        
        results = compare_experiments(configurations)
        
    elif args.compare_time_slices:
        # 比较不同时间片方法 / Compare different time slice methods
        print("\n" + "="*70)
        print("Comparing Time Slice Methods / 比较时间片方法")
        print("="*70)
        
        time_slice_types = ['rounds', 'days', 'phases', 'dynamic', 'completion']
        configurations = []
        
        for ts_type in time_slice_types:
            configurations.append({
                'name': f'{args.dataset}_{args.distribution}_{ts_type}',
                'dataset': args.dataset,
                'num_clients': args.num_clients,
                'num_rounds': args.num_rounds,
                'distribution': args.distribution,
                'time_slice_type': ts_type,
                'device': device
            })
        
        results = compare_experiments(configurations)
        
    elif args.compare_datasets:
        # 比较不同数据集 / Compare different datasets
        print("\n" + "="*70)
        print("Comparing Datasets / 比较数据集")
        print("="*70)
        
        datasets = ['mnist', 'fashion-mnist', 'cifar10']
        configurations = []
        
        for dataset in datasets:
            configurations.append({
                'name': f'{dataset}_{args.distribution}_{args.time_slice}',
                'dataset': dataset,
                'num_clients': args.num_clients,
                'num_rounds': args.num_rounds,
                'distribution': args.distribution,
                'time_slice_type': args.time_slice,
                'device': device
            })
        
        results = compare_experiments(configurations)
        
    else:
        # 运行单个实验 / Run single experiment
        print("\n" + "="*70)
        print(f"Running Experiment: {args.experiment_name}")
        print("="*70)
        print(f"Configuration / 配置:")
        print(f"  Dataset / 数据集: {args.dataset}")
        print(f"  Clients / 客户端: {args.num_clients}")
        print(f"  Rounds / 轮次: {args.num_rounds}")
        print(f"  Distribution / 分布: {args.distribution}")
        print(f"  Time Slice / 时间片: {args.time_slice}")
        
        # 创建实验运行器 / Create experiment runner
        runner = ExperimentRunner(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            distribution=args.distribution,
            time_slice_type=args.time_slice,
            device=device
        )
        
        # 运行实验 / Run experiment
        results = runner.run_experiment(args.experiment_name)
        
        print("\n" + "="*70)
        print("Experiment Complete / 实验完成")
        print("="*70)
        print(f"Results saved to: outputs/results/")
        print(f"Visualizations saved to: outputs/plots/")


if __name__ == "__main__":
    # 确保输出目录存在 / Ensure output directories exist
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    
    main()