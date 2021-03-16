"""
CNN modules.
"""
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()

        depthwise = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2, bias=bias)
        pointwise_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, padding=0, bias=bias)

        self.model = nn.Sequential(
            depthwise,
            pointwise_conv
        )

    def forward(self, x):
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
