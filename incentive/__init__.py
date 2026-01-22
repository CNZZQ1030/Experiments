"""
incentive/__init__.py - 激励模块初始化文件
Incentive Module Initialization

支持层级约束动态梯度奖励机制
Supporting Tier-Constrained Dynamic Gradient Reward Mechanism
"""

from .membership import MembershipSystem
from .points_calculator import CGSVCalculator
from .time_slice import TimeSliceManager
from .sparsification_distributor import TierConstrainedGradientDistributor, SparsificationDistributor
from .differentiated_model import UPSMDistributor, DifferentiatedModelDistributor

__all__ = [
    'MembershipSystem', 
    'CGSVCalculator', 
    'TimeSliceManager',
    'TierConstrainedGradientDistributor',
    'SparsificationDistributor',
    'UPSMDistributor',
    'DifferentiatedModelDistributor'
]