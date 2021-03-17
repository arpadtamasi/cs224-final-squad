from dataclasses import dataclass

import torch
from torch import nn as nn

from .initialized_conv import Initialized_Conv1d


@dataclass
class PerformerConf:
    nb_features: int = 256
    causal: bool = False


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout, performer_config: PerformerConf = None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)

        self.q_conv = Initialized_Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1,
            relu=False, bias=False
        )

        self.kv_conv = Initialized_Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1,
            relu=False, bias=False
        )

        from performer_pytorch import SelfAttention
        self.performer_attention = SelfAttention(
            dim=d_model,
            nb_features=performer_config.nb_features if performer_config else None,
            causal=performer_config.causal if performer_config else None
        )

        self.use_performer_attention = performer_config is not None

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, x, mask=None):
        kv = self.kv_conv(x)
        q = self.q_conv(x)

        Q = self.split_to_heads(q, self.num_head)
        K, V = [self.split_to_heads(tensor, self.num_head) for tensor in torch.split(kv, self.d_model, dim=2)]
        if self.use_performer_attention:
            x = self.performer_attention(Q, K, V)
        else:
            x = self.dot_product_attention(Q, K, V)
        return self.merge_heads(x)

    def dot_product_attention(self, q, k, v):
        q *= (self.d_model // self.num_head) ** -0.5
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, v)

    def split_to_heads(self, x, num_heads):
        batch_size, len, dim = x.shape
        return x.view(batch_size, len, num_heads, dim // num_heads).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        batch_size, num_heads, len, dim_head = x.shape
        return x.contiguous().view(batch_size, len, num_heads * dim_head)

