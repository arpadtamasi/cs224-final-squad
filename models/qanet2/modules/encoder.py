from dataclasses import dataclass

import torch
from torch import nn as nn
from torch.nn import functional as F

from .cnn import DepthwiseSeparableConv
from .initialized_conv1d import Initialized_Conv1d
from .positional_encoding import PositionalEncoding
from .self_attention import SelfAttention


@dataclass
class EncoderBlockConf:
    kernel_size: int
    layer_dropout: float
    dropout: float
    num_heads: int
    num_convs: int
    num_blocks: int


class Encoder(nn.Module):
    def __init__(self, config: EncoderBlockConf, model_dim: int):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(
                conv_num=config.num_convs,
                d_model=model_dim,
                num_head=config.num_heads,
                k=config.kernel_size,
                block_index=i, num_blocks=config.num_blocks,
                layer_dropout = config.layer_dropout,
                dropout=config.dropout
            )
            for i in range(config.num_blocks)
        ])

    def forward(self, x, mask):
        from functools import  reduce
        return reduce(lambda t, block: block(t, mask), self.blocks, x)


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, d_model, num_head, k, block_index, num_blocks, dropout=0.1, layer_dropout=0.9):
        super().__init__()
        self.layer_dropout_prob = layer_dropout
        self.positional_encoding = PositionalEncoding(d_model)

        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout
        self.block_index = block_index
        self.total_blocks = num_blocks

    def forward(self, x, mask):
        l = 0
        dropout = self.dropout
        out = self.positional_encoding(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, l)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, l)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, l)
        return out

    def layer_dropout(self, inputs, residual, sl):
        l = self.block_index * (self.conv_num + 1) + sl
        L = self.total_blocks * (self.conv_num + 1)
        prob = ((l / L) * (1 - self.layer_dropout_prob))
        if self.training == True:
            survive = torch.empty(1).uniform_(0, 1) < prob
            if survive:
                return residual
            else:
                return F.dropout(inputs, prob, training=self.training) + residual
        else:
            return inputs + residual
