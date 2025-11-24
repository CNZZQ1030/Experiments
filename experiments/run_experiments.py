#!/usr/bin/env python
"""
run_experiments_sparsification.py
实验运行脚本 - 测试不同配置的稀疏化方法
Experiment Runner Script - Test different sparsification configurations

使用方法 / Usage:
    python run_experiments_sparsification.py --experiment basic
    python run_experiments_sparsification.py --experiment comparison
    python run_experiments_sparsification.py --experiment full
"""

import subprocess
import argparse
import time
import os


def run_command(cmd):
    """运行命令并打印输出 / Run command and print output"""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def run_basic_experiments():
    """运行基础实验 / Run basic experiments"""
    experiments = [
        # MNIST IID 基准测试 / MNIST IID baseline
        {
            'name': 'MNIST_IID_Magnitude',
            'cmd': ['python', 'main_sparsification.py',
                   '--dataset', 'mnist',
                   '--distribution', 'iid',
                   '--num_clients', '10',
                   '--num_rounds', '30',
                   '--sparsification_mode', 'magnitude',
                   '--lambda_coef', '2.0']
        },
        
        # CIFAR-10 Non-IID 测试 / CIFAR-10 Non-IID test
        {
            'name': 'CIFAR10_NonIID_Magnitude',
            'cmd': ['python', 'main_sparsification.py',
                   '--dataset', 'cifar10',
                   '--distribution', 'non-iid-dir',
                   '--alpha', '0.5',
                   '--num_clients', '20',
                   '--num_rounds', '50',
                   '--sparsification_mode', 'magnitude',
                   '--lambda_coef', '2.0']
        }
    ]
    
    print("\n" + "="*80)
    print("RUNNING BASIC EXPERIMENTS")
    print("基础实验")
    print("="*80)
    
    for exp in experiments:
        print(f"\n>>> Starting experiment: {exp['name']}")
        start_time = time.time()
        
        ret_code = run_command(exp['cmd'])
        
        elapsed = time.time() - start_time
        status = "SUCCESS" if ret_code == 0 else "FAILED"
        print(f"\n>>> {exp['name']}: {status} (Time: {elapsed:.2f}s)")
        
        time.sleep(2)  # 短暂休息 / Brief pause


def run_comparison_experiments():
    """运行对比实验 - 测试不同稀疏化模式 / Run comparison experiments - test different sparsification modes"""
    base_config = {
        'dataset': 'cifar10',
        'distribution': 'non-iid-dir',
        'alpha': '0.5',
        'num_clients': '50',
        'num_rounds': '100',
        'local_epochs': '5',
        'standalone_epochs': '20'
    }
    
    experiments = [
        # 不同稀疏化模式 / Different sparsification modes
        {
            'name': 'Magnitude_Lambda2',
            'mode': 'magnitude',
            'lambda': '2.0'
        },
        {
            'name': 'Random_Lambda2',
            'mode': 'random',
            'lambda': '2.0'
        },
        {
            'name': 'Structured_Lambda2',
            'mode': 'structured',
            'lambda': '2.0'
        },
        
        # 不同Lambda值 / Different Lambda values
        {
            'name': 'Magnitude_Lambda1',
            'mode': 'magnitude',
            'lambda': '1.0'
        },
        {
            'name': 'Magnitude_Lambda3',
            'mode': 'magnitude',
            'lambda': '3.0'
        }
    ]
    
    print("\n" + "="*80)
    print("RUNNING COMPARISON EXPERIMENTS")
    print("对比实验 - 不同稀疏化配置")
    print("="*80)
    
    for exp in experiments:
        cmd = ['python', 'main_sparsification.py']
        
        # 添加基础配置 / Add base configuration
        for key, value in base_config.items():
            cmd.extend([f'--{key}', value])
        
        # 添加特定配置 / Add specific configuration
        cmd.extend(['--sparsification_mode', exp['mode']])
        cmd.extend(['--lambda_coef', exp['lambda']])
        
        print(f"\n>>> Starting experiment: {exp['name']}")
        start_time = time.time()
        
        ret_code = run_command(cmd)
        
        elapsed = time.time() - start_time
        status = "SUCCESS" if ret_code == 0 else "FAILED"
        print(f"\n>>> {exp['name']}: {status} (Time: {elapsed:.2f}s)")
        
        time.sleep(2)


def run_full_experiments():
    """运行完整实验套件 / Run full experiment suite"""
    datasets = ['mnist', 'cifar10']
    distributions = [
        {'type': 'iid', 'alpha': None},
        {'type': 'non-iid-dir', 'alpha': '0.1'},
        {'type': 'non-iid-dir', 'alpha': '0.5'}
    ]
    sparsification_modes = ['magnitude', 'structured']
    lambda_values = [1.0, 2.0, 3.0]
    
    print("\n" + "="*80)
    print("RUNNING FULL EXPERIMENT SUITE")
    print("完整实验套件")
    print("="*80)
    
    experiment_count = 0
    
    for dataset in datasets:
        for dist in distributions:
            for mode in sparsification_modes:
                for lambda_val in lambda_values:
                    experiment_count += 1
                    
                    # 构建实验名称 / Build experiment name
                    dist_name = f"{dist['type']}_a{dist['alpha']}" if dist['alpha'] else dist['type']
                    exp_name = f"{dataset}_{dist_name}_{mode}_l{lambda_val}"
                    
                    # 构建命令 / Build command
                    cmd = ['python', 'main_sparsification.py',
                          '--dataset', dataset,
                          '--distribution', dist['type'],
                          '--num_clients', '100',
                          '--num_rounds', '100',
                          '--sparsification_mode', mode,
                          '--lambda_coef', str(lambda_val),
                          '--local_epochs', '5',
                          '--standalone_epochs', '20']
                    
                    if dist['alpha']:
                        cmd.extend(['--alpha', dist['alpha']])
                    
                    print(f"\n>>> Experiment {experiment_count}: {exp_name}")
                    start_time = time.time()
                    
                    ret_code = run_command(cmd)
                    
                    elapsed = time.time() - start_time
                    status = "SUCCESS" if ret_code == 0 else "FAILED"
                    print(f"\n>>> {exp_name}: {status} (Time: {elapsed:.2f}s)")
                    
                    time.sleep(2)
    
    print(f"\n>>> Total experiments completed: {experiment_count}")


def parse_args():
    """解析命令行参数 / Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run sparsification experiments / 运行稀疏化实验'
    )
    
    parser.add_argument('--experiment', type=str, default='basic',
                       choices=['basic', 'comparison', 'full'],
                       help='Experiment type to run / 要运行的实验类型')
    
    return parser.parse_args()


def main():
    """主函数 / Main function"""
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("Sparsification-based Federated Learning Experiments")
    print("基于稀疏化的联邦学习实验")
    print(f"Experiment Type: {args.experiment}")
    print(f"{'='*80}")
    
    # 创建输出目录 / Create output directories
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    start_time = time.time()
    
    if args.experiment == 'basic':
        run_basic_experiments()
    elif args.experiment == 'comparison':
        run_comparison_experiments()
    elif args.experiment == 'full':
        run_full_experiments()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved in: outputs/results/")
    print(f"Figures saved in: outputs/figures/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()