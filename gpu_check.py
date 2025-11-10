import torch
import sys
import os
import platform
import subprocess

def check_gpu_status():
    """检查 GPU 状态的诊断函数"""
    print("=" * 50)
    print("GPU 环境诊断报告")
    print("=" * 50)
    
    # 系统信息
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检查 NVIDIA 驱动文件
    nvidia_files = [
        '/dev/nvidia0',
        '/dev/nvidia-uvm',
        '/dev/nvidiactl'
    ]
    print("\nNVIDIA 设备文件检查:")
    for file in nvidia_files:
        if os.path.exists(file):
            try:
                mode = os.stat(file).st_mode
                print(f"{file} 存在，权限: {oct(mode & 0o777)}")
            except Exception as e:
                print(f"{file} 存在，但无法获取权限: {e}")
        else:
            print(f"{file} 不存在")
    
    # CUDA 环境变量
    cuda_vars = {
        'CUDA_HOME': os.environ.get('CUDA_HOME', '未设置'),
        'CUDA_PATH': os.environ.get('CUDA_PATH', '未设置'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '未设置'),
        'NVIDIA_VISIBLE_DEVICES': os.environ.get('NVIDIA_VISIBLE_DEVICES', '未设置'),
        'NVIDIA_DRIVER_CAPABILITIES': os.environ.get('NVIDIA_DRIVER_CAPABILITIES', '未设置'),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', '未设置')
    }
    
    print("\nCUDA 环境变量:")
    for var, value in cuda_vars.items():
        print(f"{var}: {value}")
    
    # PyTorch CUDA 信息
    print("\nPyTorch CUDA 配置:")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    if hasattr(torch, '__config__'):
        print(f"PyTorch 编译配置: {torch.__config__.show()}")
    
    # 尝试获取 NVIDIA-SMI 信息
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True)
        if nvidia_smi.returncode == 0:
            print("\nNVIDIA-SMI 输出:")
            print(nvidia_smi.stdout)
        else:
            print("\nNVIDIA-SMI 执行失败")
            print(f"错误信息: {nvidia_smi.stderr}")
    except Exception as e:
        print("\nNVIDIA-SMI 未找到或无法执行")
        print(f"错误信息: {str(e)}")
    
    if not torch.cuda.is_available():
        print("\nCUDA 不可用的可能原因:")
        print("1. CUDA 环境变量未正确设置")
        print("2. PyTorch CUDA 版本与系统不匹配")
        print("3. NVIDIA 驱动权限问题")
        
        print("\n建议解决步骤:")
        print("1. 设置正确的环境变量")
        print("2. 检查 NVIDIA 驱动权限")
        print("3. 重新安装匹配的 PyTorch CUDA 版本")
    
    print("\n诊断完成")
    print("=" * 50)

if __name__ == "__main__":
    check_gpu_status()