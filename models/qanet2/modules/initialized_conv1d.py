from torch import nn as nn


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.activation = nn.ReLU()
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.activation = nn.Identity()
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        return self.activation(self.out(x))
