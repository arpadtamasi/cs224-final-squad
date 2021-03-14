import torch
from torch import nn as nn

from .initialized_conv import Initialized_Conv1d


class HighwayNetwork(nn.Module):
    def __init__(self, dropout, hidden_size, num_layers):
        super().__init__()
        self.model = nn.Sequential(*(
            HighwayLayer(dropout, hidden_size)
            for _ in range(num_layers)
        ))

    def forward(self, x):
        return self.model(x)

class HighwayLayer(nn.Module):
    def __init__(self, dropout, hidden_size):
        super(HighwayLayer, self).__init__()

        self.transform = Initialized_Conv1d(hidden_size, hidden_size, bias=True, relu=False)
        self.gate = Initialized_Conv1d(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        t = self.dropout(self.transform(x))
        return g * t + (1 - g) * x
