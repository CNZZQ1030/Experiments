"""
federated/server.py
服务器类定义 - 支持基于贡献度的差异化模型分发
Server Class Definition - Supporting Contribution-based Differentiated Model Distribution
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


class DifferentiatedModelDistributor:
    """
    差异化模型分发器 / Differentiated Model Distributor
    根据客户端贡献度生成个性化模型
    Generate personalized models based on client contributions
    """
    
    def __init__(self):
        """初始化分发器 / Initialize distributor"""
        self.contribution_thresholds = {
            'diamond': 0.75,  # 贡献度 > 75% 获得钻石级模型
            'gold': 0.50,     # 贡献度 > 50% 获得金级模型
            'silver': 0.25,   # 贡献度 > 25% 获得银级模型
            'bronze': 0.0     # 贡献度 >= 0% 获得铜级模型
        }
        
    def create_differentiated_model(self, base_model_weights: Dict,
                                   client_updates: Dict[int, Dict],
                                   client_contributions: Dict[int, float],
                                   target_client_id: int,
                                   target_contribution: float) -> Dict:
        """
        为特定客户端创建差异化模型 / Create differentiated model for specific client
        
        Args:
            base_model_weights: 基础模型权重（客户端本地模型）/ Base model weights (client local model)
            client_updates: 所有客户端的模型更新 / All client model updates
            client_contributions: 客户端贡献度字典 / Client contribution dictionary
            target_client_id: 目标客户端ID / Target client ID
            target_contribution: 目标客户端的贡献度 / Target client contribution
            
        Returns:
            差异化模型权重 / Differentiated model weights
        """
        # 如果贡献度为0，只返回客户端自己的本地模型 / If contribution is 0, return only local model
        if target_contribution == 0:
            if target_client_id in client_updates:
                return client_updates[target_client_id]
            else:
                return base_model_weights
        
        # 根据贡献度确定可以获得多少其他客户端的更新 / Determine how many updates based on contribution
        # 贡献度越高，可以获得越多高质量客户端的更新 / Higher contribution gets more high-quality updates
        
        # 对客户端按贡献度排序 / Sort clients by contribution
        sorted_clients = sorted(client_contributions.items(), 
                              key=lambda x: x[1], reverse=True)
        
        # 计算可以访问的客户端数量 / Calculate number of accessible clients
        num_total_clients = len(sorted_clients)
        num_accessible = max(1, int(target_contribution * num_total_clients))
        
        # 选择可访问的客户端 / Select accessible clients
        accessible_clients = [cid for cid, _ in sorted_clients[:num_accessible]]
        
        # 确保目标客户端自己总是包含在内 / Ensure target client is always included
        if target_client_id not in accessible_clients:
            accessible_clients.append(target_client_id)
        
        # 聚合可访问客户端的模型 / Aggregate accessible client models
        aggregated_weights = {}
        total_weight = 0
        
        for client_id in accessible_clients:
            if client_id not in client_updates:
                continue
            
            # 根据贡献度分配权重 / Assign weights based on contribution
            client_weight = client_contributions.get(client_id, 0.1)
            
            # 如果是目标客户端自己，给予额外权重 / Give extra weight to target client itself
            if client_id == target_client_id:
                client_weight *= 1.5
            
            weights = client_updates[client_id]
            
            for key in weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key])
                
                aggregated_weights[key] += weights[key] * client_weight
            
            total_weight += client_weight
        
        # 归一化 / Normalize
        if total_weight > 0:
            for key in aggregated_weights.keys():
                aggregated_weights[key] /= total_weight
        
        return aggregated_weights


class FederatedServer:
    """
    联邦学习服务器 / Federated Learning Server
    负责模型聚合和基于贡献度的差异化模型分发
    Responsible for model aggregation and contribution-based differentiated model distribution
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")):
        """
        初始化服务器 / Initialize server
        
        Args:
            model: 全局模型 / Global model
            device: 计算设备 / Computing device
        """
        self.global_model = copy.deepcopy(model).to(device)
        self.device = device
        
        # 模型分发器 / Model distributor
        self.model_distributor = DifferentiatedModelDistributor()
        
        # 训练历史 / Training history
        self.accuracy_history = []
        self.loss_history = []
        self.round_times = []
        
        # 客户端信息和贡献度 / Client information and contributions
        self.client_info = {}
        self.client_contributions = {}
        self.client_gradients = {}
        
        # 存储每个客户端的个性化模型 / Store personalized models for each client
        self.client_models = {}
        
        # 全局梯度 / Global gradient
        self.global_gradient = None
        self.prev_global_weights = None
        
    def get_global_weights(self) -> Dict:
        """
        获取全局模型权重 / Get global model weights
        
        Returns:
            全局模型权重字典 / Global model weights dictionary
        """
        return self.global_model.state_dict()
    
    def get_client_model_weights(self, client_id: int) -> Dict:
        """
        获取特定客户端的个性化模型权重 / Get personalized model weights for specific client
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            个性化模型权重 / Personalized model weights
        """
        if client_id in self.client_models:
            return self.client_models[client_id]
        else:
            # 如果还没有个性化模型，返回全局模型 / Return global model if no personalized model yet
            return self.global_model.state_dict()
    
    def compute_gradients(self, client_weights: Dict[int, Dict]) -> Dict[int, np.ndarray]:
        """
        计算客户端梯度 / Compute client gradients
        
        Args:
            client_weights: 客户端模型权重 / Client model weights
            
        Returns:
            客户端梯度字典 / Client gradients dictionary
        """
        gradients = {}
        
        if self.prev_global_weights is None:
            self.prev_global_weights = self.get_global_weights()
        
        for client_id, weights in client_weights.items():
            gradient_vectors = []
            
            for key in weights.keys():
                if key in self.prev_global_weights:
                    # 计算梯度 / Calculate gradient
                    grad = weights[key] - self.prev_global_weights[key]
                    gradient_vectors.append(grad.cpu().numpy().flatten())
            
            # 连接为一个向量 / Concatenate into one vector
            gradients[client_id] = np.concatenate(gradient_vectors)
        
        return gradients
    
    def aggregate_models(self, client_weights: Dict[int, Dict], 
                        client_infos: Dict[int, Dict],
                        client_contributions: Dict[int, float]) -> None:
        """
        聚合客户端模型并生成差异化模型 / Aggregate client models and generate differentiated models
        
        Args:
            client_weights: 客户端模型权重 / Client model weights
            client_infos: 客户端训练信息 / Client training information
            client_contributions: 客户端贡献度 / Client contributions
        """
        # 保存当前全局权重作为前一轮权重 / Save current global weights as previous weights
        self.prev_global_weights = copy.deepcopy(self.get_global_weights())
        
        # 保存客户端贡献度 / Save client contributions
        self.client_contributions = client_contributions
        
        # 标准FedAvg聚合生成全局模型 / Standard FedAvg aggregation for global model
        self._fedavg_aggregation(client_weights, client_infos)
        
        # 计算全局梯度 / Calculate global gradient
        global_gradient_vectors = []
        new_global_weights = self.get_global_weights()
        
        for key in new_global_weights.keys():
            if key in self.prev_global_weights:
                grad = new_global_weights[key] - self.prev_global_weights[key]
                global_gradient_vectors.append(grad.cpu().numpy().flatten())
        
        self.global_gradient = np.concatenate(global_gradient_vectors)
        
        # 为每个参与的客户端生成差异化模型 / Generate differentiated models for each participating client
        for client_id in client_weights.keys():
            contribution = client_contributions.get(client_id, 0)
            
            # 创建个性化模型 / Create personalized model
            personalized_weights = self.model_distributor.create_differentiated_model(
                base_model_weights=client_weights[client_id],
                client_updates=client_weights,
                client_contributions=client_contributions,
                target_client_id=client_id,
                target_contribution=contribution
            )
            
            self.client_models[client_id] = personalized_weights
    
    def _fedavg_aggregation(self, client_weights: Dict[int, Dict], 
                           client_infos: Dict[int, Dict]) -> None:
        """
        FedAvg聚合算法 / FedAvg aggregation algorithm
        
        Args:
            client_weights: 客户端模型权重 / Client model weights
            client_infos: 客户端信息 / Client information
        """
        total_samples = sum(info['num_samples'] for info in client_infos.values())
        aggregated_weights = {}
        
        for client_id, weights in client_weights.items():
            client_samples = client_infos[client_id]['num_samples']
            weight_factor = client_samples / total_samples if total_samples > 0 else 0
            
            for key in weights.keys():
                if key not in aggregated_weights:
                    aggregated_weights[key] = torch.zeros_like(weights[key])
                
                aggregated_weights[key] += weights[key] * weight_factor
        
        self.global_model.load_state_dict(aggregated_weights)
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        评估全局模型 / Evaluate global model
        
        Args:
            test_loader: 测试数据加载器 / Test data loader
            
        Returns:
            准确率和损失 / Accuracy and loss
        """
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        self.accuracy_history.append(accuracy)
        self.loss_history.append(avg_loss)
        
        return accuracy, avg_loss
    
    def evaluate_client_models(self, test_loader: DataLoader, 
                              client_ids: List[int]) -> Dict[int, Tuple[float, float]]:
        """
        评估客户端个性化模型 / Evaluate client personalized models
        
        Args:
            test_loader: 测试数据加载器 / Test data loader
            client_ids: 要评估的客户端ID列表 / Client IDs to evaluate
            
        Returns:
            客户端模型性能字典 / Client model performance dictionary
        """
        results = {}
        criterion = nn.CrossEntropyLoss()
        
        for client_id in client_ids:
            if client_id not in self.client_models:
                continue
            
            # 创建临时模型用于评估 / Create temporary model for evaluation
            temp_model = copy.deepcopy(self.global_model)
            temp_model.load_state_dict(self.client_models[client_id])
            temp_model.eval()
            
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = temp_model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += len(data)
            
            accuracy = correct / total if total > 0 else 0
            avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
            
            results[client_id] = (accuracy, avg_loss)
        
        return results
    
    def get_contribution_based_model_quality(self, client_id: int) -> float:
        """
        获取基于贡献度的模型质量评分 / Get contribution-based model quality score
        
        Args:
            client_id: 客户端ID / Client ID
            
        Returns:
            模型质量评分 / Model quality score
        """
        contribution = self.client_contributions.get(client_id, 0)
        
        # 根据贡献度计算预期模型质量 / Calculate expected model quality based on contribution
        # 使用非线性映射，贡献度越高，质量提升越明显 / Non-linear mapping for quality
        quality_score = 0.3 + 0.7 * (contribution ** 0.5)
        
        return quality_score
    
    def save_checkpoint(self, round_num: int, save_path: str) -> None:
        """
        保存检查点 / Save checkpoint
        
        Args:
            round_num: 当前轮次 / Current round
            save_path: 保存路径 / Save path
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'client_contributions': self.client_contributions,
            'client_models_sample': {k: v for k, v in list(self.client_models.items())[:5]}  # 保存部分客户端模型作为样本
        }
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at round {round_num}: {save_path}")