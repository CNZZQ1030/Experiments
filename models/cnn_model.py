"""
models/cnn_model.py (Extended Version with ResNet)
模型定义 - 扩展版本，增加ResNet支持
Model Definition - Extended Version with ResNet Support

支持的模型 / Supported Models:
- SimpleCNN: 用于MNIST, Fashion-MNIST / For MNIST, Fashion-MNIST
- CIFARCNN: 用于CIFAR-10 / For CIFAR-10
- ResNet18/34: 用于CIFAR-100（更高级的模型）/ For CIFAR-100 (advanced model)
- TextCNN: 用于SST文本分类 / For SST text classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


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
    适用于CIFAR-10 / Suitable for CIFAR-10
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


# =====================================
# ResNet实现 - 适用于CIFAR-100
# ResNet Implementation - For CIFAR-100
# =====================================

class BasicBlock(nn.Module):
    """
    ResNet基础块 / ResNet Basic Block
    用于ResNet18和ResNet34 / For ResNet18 and ResNet34
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    ResNet瓶颈块 / ResNet Bottleneck Block
    用于ResNet50及更深的网络 / For ResNet50 and deeper
    """
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetCIFAR(nn.Module):
    """
    适用于CIFAR数据集的ResNet / ResNet for CIFAR datasets
    
    与标准ResNet的区别 / Differences from standard ResNet:
    1. 第一层使用3x3卷积而非7x7 / First layer uses 3x3 conv instead of 7x7
    2. 没有初始的max pooling / No initial max pooling
    3. 适配32x32的小图像 / Adapted for 32x32 small images
    """
    
    def __init__(self, block, layers: List[int], num_classes: int = 100,
                 input_channels: int = 3, zero_init_residual: bool = False):
        """
        初始化ResNet / Initialize ResNet
        
        Args:
            block: 基础块类型 (BasicBlock 或 Bottleneck)
            layers: 每个阶段的块数量 / Number of blocks per stage
            num_classes: 分类数量 / Number of classes
            input_channels: 输入通道数 / Input channels
            zero_init_residual: 是否初始化残差分支BN为零 / Zero-init residual BN
        """
        super(ResNetCIFAR, self).__init__()
        
        self.in_channels = 64
        
        # 初始卷积层 - 适配CIFAR的小图像
        # Initial conv layer - adapted for CIFAR's small images
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 注意：没有max pooling / Note: no max pooling
        
        # 残差层 / Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化和分类器 / Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重 / Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """
        创建残差层 / Create residual layer
        
        Args:
            block: 块类型 / Block type
            out_channels: 输出通道数 / Output channels
            blocks: 块数量 / Number of blocks
            stride: 步长 / Stride
            
        Returns:
            Sequential模块 / Sequential module
        """
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 / Forward pass"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def ResNet18(num_classes: int = 100, input_channels: int = 3) -> ResNetCIFAR:
    """
    创建ResNet18模型 / Create ResNet18 model
    适合CIFAR-100 / Suitable for CIFAR-100
    
    Args:
        num_classes: 分类数量 / Number of classes
        input_channels: 输入通道数 / Input channels
        
    Returns:
        ResNet18模型 / ResNet18 model
    """
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)


def ResNet34(num_classes: int = 100, input_channels: int = 3) -> ResNetCIFAR:
    """
    创建ResNet34模型 / Create ResNet34 model
    
    Args:
        num_classes: 分类数量 / Number of classes
        input_channels: 输入通道数 / Input channels
        
    Returns:
        ResNet34模型 / ResNet34 model
    """
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes, input_channels)


def ResNet50(num_classes: int = 100, input_channels: int = 3) -> ResNetCIFAR:
    """
    创建ResNet50模型 / Create ResNet50 model
    
    Args:
        num_classes: 分类数量 / Number of classes
        input_channels: 输入通道数 / Input channels
        
    Returns:
        ResNet50模型 / ResNet50 model
    """
    return ResNetCIFAR(Bottleneck, [3, 4, 6, 3], num_classes, input_channels)


# =====================================
# VGG实现 - 作为备选的高级模型
# VGG Implementation - As alternative advanced model
# =====================================

class VGG11(nn.Module):
    """
    VGG11模型（适配CIFAR）/ VGG11 model (adapted for CIFAR)
    作为ResNet的备选 / As alternative to ResNet
    """
    
    def __init__(self, num_classes: int = 100, input_channels: int = 3):
        super(VGG11, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =====================================
# 文本分类模型
# Text Classification Models
# =====================================

class TextCNN(nn.Module):
    """
    用于文本分类的TextCNN模型 / TextCNN Model for text classification
    适用于SST情感分析任务 / Suitable for SST sentiment analysis task
    """
    
    def __init__(self, num_classes: int = 2, vocab_size: int = 20000,
                 embedding_dim: int = 128, filter_sizes: List[int] = [3, 4, 5],
                 num_filters: int = 100, dropout: float = 0.5,
                 max_seq_length: int = 200):
        super(TextCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class TextLSTM(nn.Module):
    """
    用于文本分类的LSTM模型 / LSTM Model for text classification
    """
    
    def __init__(self, num_classes: int = 2, vocab_size: int = 20000,
                 embedding_dim: int = 128, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.5,
                 bidirectional: bool = True):
        super(TextLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        out = self.dropout(hidden)
        out = self.fc(out)
        
        return out


# =====================================
# 模型工厂
# Model Factory
# =====================================

class ModelFactory:
    """
    模型工厂类，用于创建不同的模型实例
    Model factory class for creating different model instances
    
    重要更新：CIFAR-100现在使用ResNet18
    Important update: CIFAR-100 now uses ResNet18
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
        input_channels = kwargs.get('input_channels', 3)
        
        # MNIST和Fashion-MNIST使用SimpleCNN
        if model_name in ['simple_cnn', 'mnist', 'fashion-mnist', 'fashion_mnist']:
            input_channels = kwargs.get('input_channels', 1)
            return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
        
        # CIFAR-10使用CIFARCNN
        elif model_name in ['cifar_cnn', 'cifar10']:
            input_channels = kwargs.get('input_channels', 3)
            return CIFARCNN(num_classes=num_classes, input_channels=input_channels)
        
        # CIFAR-100使用ResNet18（重要更新！）
        elif model_name in ['cifar100']:
            print(f"[ModelFactory] Using ResNet18 for CIFAR-100 (100 classes)")
            return ResNet18(num_classes=num_classes, input_channels=input_channels)
        
        # 直接指定ResNet类型
        elif model_name in ['resnet18']:
            return ResNet18(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name in ['resnet34']:
            return ResNet34(num_classes=num_classes, input_channels=input_channels)
        
        elif model_name in ['resnet50']:
            return ResNet50(num_classes=num_classes, input_channels=input_channels)
        
        # VGG11
        elif model_name in ['vgg11', 'vgg']:
            return VGG11(num_classes=num_classes, input_channels=input_channels)
        
        # SST使用TextCNN
        elif model_name in ['textcnn', 'sst']:
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
        
        # TextLSTM
        elif model_name in ['textlstm', 'lstm']:
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
                'model_class': 'ResNet18',  # 更新为ResNet18
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
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """
        统计模型参数数量 / Count model parameters
        
        Args:
            model: 模型 / Model
            
        Returns:
            可训练参数数量 / Number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def print_model_summary(model: nn.Module, model_name: str = ""):
        """
        打印模型摘要 / Print model summary
        
        Args:
            model: 模型 / Model
            model_name: 模型名称 / Model name
        """
        total_params = ModelFactory.count_parameters(model)
        print(f"\n{'='*50}")
        print(f"Model Summary: {model_name}")
        print(f"{'='*50}")
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Approximate size: {total_params * 4 / (1024**2):.2f} MB (float32)")
        print(f"{'='*50}")