"""
federated/__init__.py - 联邦学习模块初始化文件
Federated Learning Module Initialization
"""

from .client import FederatedClient
from .server import FederatedServerWithGradientSparsification

__all__ = ['FederatedClient', 'FederatedServerWithGradientSparsification']