import torch
from torch import nn as nn
from torch.nn import functional as F

from .depthwise_separable_conv import DepthwiseSeparableConv
from .highway import Highway


class Embedding(nn.Module):
    def __init__(self, d_word, d_char, dropout, dropout_char):
        super(Embedding, self).__init__()
        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2)
        self.high = Highway(2, d_word + d_char)
        self.dropout = dropout
        self.dropout_char = dropout_char

    def forward(self, wd_emb, ch_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=-1)
        # ch_emb = ch_emb.squeeze()
        wd_emb = F.dropout(wd_emb, p=self.dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.high(emb)
        return emb
