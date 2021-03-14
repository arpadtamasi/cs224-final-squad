import typing
from dataclasses import dataclass

import torch
from torch import nn as nn

from .cnn import DepthwiseSeparableConv
from .initialized_conv import Initialized_Conv1d
from .positional_encoding import PositionalEncoding
from .self_attention import SelfAttention, PerformerConf


@dataclass
class EncoderBlockConf:
    kernel_size: int
    layer_dropout: float
    dropout: float
    num_heads: int
    num_convs: int
    num_blocks: int

    performer: typing.Optional[PerformerConf] = None


class Encoder(nn.Module):
    def __init__(self, config: EncoderBlockConf, model_dim: int, use_performer=False):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(
                config=config,
                model_dim=model_dim,
                block_index=i,
                use_performer=use_performer
            )
            for i in range(config.num_blocks)
        ])

    def forward(self, x, mask):
        from functools import reduce
        return reduce(lambda t, block: block(t, mask), self.blocks, x)


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderBlockConf, model_dim: int, block_index: int, use_performer=False):
        super().__init__()

        self.positional_encoding = PositionalEncoding(model_dim)

        layers_per_block = (config.num_convs + 2)
        last_layer = config.num_blocks * layers_per_block
        first_layer = block_index * layers_per_block
        l = first_layer
        self.conv_blocks = nn.Sequential(*[
            ResidualConnection(
                model_dim=model_dim,
                submodule=DepthwiseSeparableConv(model_dim, model_dim, config.kernel_size),
                l=l + i, L=last_layer, layer_dropout=config.layer_dropout,
                layernorm=True, dropout=config.dropout
            )
            for i in range(config.num_convs)
        ])

        l = first_layer + config.num_blocks + 1
        self.attention = ResidualConnection(
            model_dim=model_dim,
            submodule=SelfAttention(model_dim, config.num_heads, dropout=config.dropout, performer_config=config.performer),
            l=l, L=last_layer, layer_dropout=config.layer_dropout,
            layernorm=True, dropout=config.dropout
        )

        l = first_layer + config.num_blocks + 2
        self.feedforward = ResidualConnection(
            model_dim=model_dim,
            submodule=nn.Sequential(
                Initialized_Conv1d(model_dim, model_dim, relu=True, bias=True),
                Initialized_Conv1d(model_dim, model_dim, relu=False, bias=True)
            ),
            l=l, L=last_layer, layer_dropout=config.layer_dropout,
            layernorm=True, dropout=config.dropout
        )

    def forward(self, x, mask):
        enc = self.positional_encoding(x)
        conv = self.conv_blocks(enc)
        att = self.attention(conv, mask=mask)
        ff = self.feedforward(att)
        return ff


class ResidualConnection(nn.Module):
    def __init__(self, model_dim: int, submodule: nn.Module, l: int, L: int, layer_dropout: float, dropout: float, layernorm: bool = False):
        super(ResidualConnection, self).__init__()

        dropout_prob = 1 - ((l / L) * (1 - layer_dropout))
        self.survival = torch.FloatTensor([dropout_prob])
        self.norm = nn.LayerNorm(model_dim) if layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.submodule = submodule

    def forward(self, x, **kwargs):
        drop = (
                self.training
                and self.survival is not None
                and torch.bernoulli(self.survival).item() == 0
        )
        if drop:
            return x
        else:
            normalized = self.norm(x.transpose(1, 2)).transpose(1, 2)
            normalized = self.dropout(normalized)
            transformed = self.submodule(normalized, **kwargs)
            return x + transformed
