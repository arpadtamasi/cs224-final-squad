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
    def __init__(self, config: EncoderBlockConf, model_dim: int, use_performer=False):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(
                config = config,
                model_dim = model_dim,
                block_index=i,
                use_performer=use_performer
            )
            for i in range(config.num_blocks)
        ])

    def forward(self, x, mask):
        from functools import  reduce
        return reduce(lambda t, block: block(t, mask), self.blocks, x)


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderBlockConf, model_dim: int, block_index: int, use_performer = False):
        super().__init__()
        self.dropout_prob = config.dropout
        self.layer_dropout_prob = config.layer_dropout
        self.positional_encoding = PositionalEncoding(model_dim)
        self.conv_num = config.num_convs

        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(model_dim, model_dim, config.kernel_size)
            for _ in range(config.num_convs)
        ])

        self.self_att = SelfAttention(model_dim, config.num_heads, dropout=config.dropout, use_performer=use_performer)

        self.FFN_1 = Initialized_Conv1d(model_dim, model_dim, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(model_dim, model_dim, bias=True)
        self.norm_C = nn.ModuleList([
            nn.LayerNorm(model_dim)
            for _ in range(config.num_convs)
        ])
        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)
        self.block_index = block_index
        self.total_blocks = config.num_blocks

    def forward(self, x, mask):
        l = 0
        dropout = self.dropout_prob
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
        out = self.self_att(out, mask=mask)
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
