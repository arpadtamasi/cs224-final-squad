from torch import nn as nn
import torch

from .initialized_conv1d import Initialized_Conv1d


class Highway(nn.Module):
    def __init__(self, dropout, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])
        self.dropout = nn.Dropout(dropout)
        self.children()

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        dropout = 0.1
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = self.dropout(nonlinear)
            x = gate * nonlinear + (1 - gate) * x
        return x
