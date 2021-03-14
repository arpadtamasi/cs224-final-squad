from typing import Tuple

from torch import nn as nn


class Initialized_Conv1d(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int = 1, stride: int = 1,
                 padding: int = 0, groups: int = 1,
                 relu: bool = False, bias: bool = False
                 ):
        super().__init__()

        transform = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias)

        if relu is True:
            nn.init.kaiming_normal_(transform.weight, nonlinearity='relu')
            self.model = nn.Sequential(transform, nn.ReLU())
        else:
            nn.init.xavier_uniform_(transform.weight)
            self.model = transform

    def forward(self, x):
        return self.model(x)


class Initialized_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int] = 1, stride: int = 1,
                 padding: int = 0, groups: int = 1,
                 relu: bool = False, bias: bool = False
                 ):
        super().__init__()

        transform = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias)

        if relu is True:
            nn.init.kaiming_normal_(transform.weight, nonlinearity='relu')
            self.model = nn.Sequential(transform, nn.ReLU())
        else:
            nn.init.xavier_uniform_(transform.weight)
            self.model = transform

    def forward(self, x):
        return self.model(x)
