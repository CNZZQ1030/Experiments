"""
models/cnn_model.py (Extended Version)
模型定义 - 扩展版本 / Model Definition - Extended Version

支持的模型 / Supported Models:
- SimpleCNN: 用于MNIST, Fashion-MNIST / For MNIST, Fashion-MNIST
- CIFARCNN: 用于CIFAR-10, CIFAR-100 / For CIFAR-10, CIFAR-100
- TextCNN: 用于SST文本分类 / For SST text classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SimpleCNN(nn.Module):
    """
    简单的CNN模型 / Simple CNN Model
    适用于MNIST和Fashion-MNIST数据集
    Suitable for MNIST and Fashion-MNIST datasets
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        """
        初始化模型 / Initialize model
        
        Args:
            num_classes: 分类数量 / Number of classes
            input_channels: 输入通道数 / Number of input channels
        """
        super(SimpleCNN, self).__init__()
        
        # 卷积层 / Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 批归一化层 / Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层 / Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层 / Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout层 / Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass
        
        Args:
            x: 输入张量 / Input tensor
            
        Returns:
            输出张量 / Output tensor
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CIFARCNN(nn.Module):
    """
    用于CIFAR数据集的CNN模型 / CNN Model for CIFAR datasets
    适用于CIFAR-10和CIFAR-100 / Suitable for CIFAR-10 and CIFAR-100
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        初始化模型 / Initialize model
        
        Args:
            num_classes: 分类数量 / Number of classes
            input_channels: 输入通道数 / Number of input channels
        """
        super(CIFARCNN, self).__init__()
        
        # 卷积块1 / Convolution block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积块2 / Convolution block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 全连接层 / Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass
        
        Args:
            x: 输入张量 / Input tensor
            
        Returns:
            输出张量 / Output tensor
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TextCNN(nn.Module):
    """
    用于文本分类的TextCNN模型 / TextCNN Model for text classification
    适用于SST情感分析任务 / Suitable for SST sentiment analysis task
    
    使用多种卷积核大小捕获不同长度的n-gram特征
    Uses multiple filter sizes to capture n-gram features of different lengths
    """
    
    def __init__(self, num_classes: int = 2, vocab_size: int = 20000,
                 embedding_dim: int = 128, filter_sizes: List[int] = [3, 4, 5],
                 num_filters: int = 100, dropout: float = 0.5,
                 max_seq_length: int = 200):
        """
        初始化TextCNN模型 / Initialize TextCNN model
        
        Args:
            num_classes: 分类数量 / Number of classes
            vocab_size: 词汇表大小 / Vocabulary size
            embedding_dim: 词嵌入维度 / Word embedding dimension
            filter_sizes: 卷积核大小列表 / List of filter sizes
            num_filters: 每种大小的卷积核数量 / Number of filters per size
            dropout: Dropout率 / Dropout rate
            max_seq_length: 最大序列长度 / Maximum sequence length
        """
        super(TextCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # 词嵌入层 / Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 卷积层 / Convolutional layers
        # 对于文本，使用1D卷积 / For text, use 1D convolution
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # 全连接层 / Fully connected layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        # Dropout层 / Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass
        
        Args:
            x: 输入张量 [batch_size, seq_length] / Input tensor
            
        Returns:
            输出张量 [batch_size, num_classes] / Output tensor
        """
        # 词嵌入 / Word embedding
        # x: [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
        x = self.embedding(x)
        
        # 转置以适应Conv1d / Transpose for Conv1d
        # [batch_size, seq_length, embedding_dim] -> [batch_size, embedding_dim, seq_length]
        x = x.permute(0, 2, 1)
        
        # 多尺度卷积 / Multi-scale convolution
        conv_outputs = []
        for conv in self.convs:
            # 卷积 + ReLU / Convolution + ReLU
            conv_out = F.relu(conv(x))
            # 最大池化 / Max pooling over time
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # 拼接所有卷积输出 / Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)
        
        # Dropout和全连接 / Dropout and fully connected
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class TextLSTM(nn.Module):
    """
    用于文本分类的LSTM模型 / LSTM Model for text classification
    作为TextCNN的替代选择 / Alternative to TextCNN
    """
    
    def __init__(self, num_classes: int = 2, vocab_size: int = 20000,
                 embedding_dim: int = 128, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.5,
                 bidirectional: bool = True):
        """
        初始化LSTM模型 / Initialize LSTM model
        
        Args:
            num_classes: 分类数量 / Number of classes
            vocab_size: 词汇表大小 / Vocabulary size
            embedding_dim: 词嵌入维度 / Word embedding dimension
            hidden_size: 隐藏层大小 / Hidden layer size
            num_layers: LSTM层数 / Number of LSTM layers
            dropout: Dropout率 / Dropout rate
            bidirectional: 是否双向 / Whether bidirectional
        """
        super(TextLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 词嵌入层 / Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM层 / LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 全连接层 / Fully connected layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        
        # Dropout层 / Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 / Forward pass
        
        Args:
            x: 输入张量 [batch_size, seq_length] / Input tensor
            
        Returns:
            输出张量 [batch_size, num_classes] / Output tensor
        """
        # 词嵌入 / Word embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 获取最后一个时间步的输出 / Get output of last time step
        if self.bidirectional:
            # 拼接双向的最后隐藏状态 / Concatenate last hidden states from both directions
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Dropout和全连接 / Dropout and fully connected
        out = self.dropout(hidden)
        out = self.fc(out)
        
        return out


class ModelFactory:
    """
    模型工厂类，用于创建不同的模型实例
    Model factory class for creating different model instances
    """
    
    @staticmethod
    def create_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
        """
        创建模型实例 / Create model instance
        
        Args:
            model_name: 模型名称或数据集名称 / Model name or dataset name
            num_classes: 分类数量 / Number of classes
            **kwargs: 其他参数 / Other parameters
            
        Returns:
            模型实例 / Model instance
        """
        model_name = model_name.lower()
        kwargs['num_classes'] = num_classes
        
        # MNIST和Fashion-MNIST使用SimpleCNN / Use SimpleCNN for MNIST and Fashion-MNIST
        if model_name in ['simple_cnn', 'mnist', 'fashion-mnist', 'fashion_mnist']:
            kwargs['input_channels'] = kwargs.get('input_channels', 1)
            return SimpleCNN(**kwargs)
        
        # CIFAR-10和CIFAR-100使用CIFARCNN / Use CIFARCNN for CIFAR-10 and CIFAR-100
        elif model_name in ['cifar_cnn', 'cifar10', 'cifar100']:
            kwargs['input_channels'] = kwargs.get('input_channels', 3)
            return CIFARCNN(**kwargs)
        
        # SST使用TextCNN / Use TextCNN for SST
        elif model_name in ['textcnn', 'sst']:
            # 移除不适用于TextCNN的参数 / Remove parameters not applicable to TextCNN
            kwargs.pop('input_channels', None)
            
            # 设置TextCNN默认参数 / Set TextCNN default parameters
            vocab_size = kwargs.pop('vocab_size', 20000)
            embedding_dim = kwargs.pop('embedding_dim', 128)
            filter_sizes = kwargs.pop('filter_sizes', [3, 4, 5])
            num_filters = kwargs.pop('num_filters', 100)
            dropout = kwargs.pop('dropout', 0.5)
            max_seq_length = kwargs.pop('max_seq_length', 200)
            
            return TextCNN(
                num_classes=num_classes,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                dropout=dropout,
                max_seq_length=max_seq_length
            )
        
        # TextLSTM作为备选 / TextLSTM as alternative
        elif model_name in ['textlstm', 'lstm']:
            kwargs.pop('input_channels', None)
            
            vocab_size = kwargs.pop('vocab_size', 20000)
            embedding_dim = kwargs.pop('embedding_dim', 128)
            hidden_size = kwargs.pop('hidden_size', 128)
            num_layers = kwargs.pop('num_layers', 2)
            dropout = kwargs.pop('dropout', 0.5)
            bidirectional = kwargs.pop('bidirectional', True)
            
            return TextLSTM(
                num_classes=num_classes,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        
        else:
            raise ValueError(f"Unknown model/dataset name: {model_name}")
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """
        获取模型信息 / Get model information
        
        Args:
            model_name: 模型名称 / Model name
            
        Returns:
            模型信息字典 / Model information dictionary
        """
        model_name = model_name.lower()
        
        info = {
            'mnist': {
                'model_class': 'SimpleCNN',
                'input_channels': 1,
                'input_size': (28, 28),
                'num_classes': 10
            },
            'fashion-mnist': {
                'model_class': 'SimpleCNN',
                'input_channels': 1,
                'input_size': (28, 28),
                'num_classes': 10
            },
            'cifar10': {
                'model_class': 'CIFARCNN',
                'input_channels': 3,
                'input_size': (32, 32),
                'num_classes': 10
            },
            'cifar100': {
                'model_class': 'CIFARCNN',
                'input_channels': 3,
                'input_size': (32, 32),
                'num_classes': 100
            },
            'sst': {
                'model_class': 'TextCNN',
                'vocab_size': 20000,
                'embedding_dim': 128,
                'max_seq_length': 200,
                'num_classes': 2
            }
        }
        
        return info.get(model_name, {})