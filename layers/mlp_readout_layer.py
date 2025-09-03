import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=number of hidden layers
        super().__init__()
        
        # 创建隐藏层 + Layer Normalization
        self.FC_layers = nn.ModuleList()
        self.LN_layers = nn.ModuleList()

        # 为每个隐藏层添加 FC 和 LN 层
        for l in range(L):
            # 添加全连接层
            self.FC_layers.append(nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=False))
            # 添加层归一化 (Layer Normalization)
            self.LN_layers.append(nn.LayerNorm(input_dim // 2 ** (l + 1)))
        
        # 输出层
        self.FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=False))
        
        # Dropout 层
        self.dropout = nn.Dropout(0.3)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            # 当前层的输出
            y = self.FC_layers[l](y)
            # 应用 Layer Normalization
            y = self.LN_layers[l](y)
            # 激活函数
            y = F.gelu(y)
            # Dropout
            y = self.dropout(y)

        # 最后一层，不使用 Dropout 和 Layer Normalization
        y = self.FC_layers[self.L](y)
        return y