import copy
import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def __init__(self, dropout=0.0):
        """
        :param dropout: attention dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        p_attn = self.softmax(scores)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Compute 'Multi-Head Attention'
    When we calculate attentions, usually key and value are the same tensor.
    For self-attention, query, key, value are all the same tensor.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of heads
        :param d_model: hidden size
        :param dropout: attention dropout rate
        """
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = ScaledDotProductAttention(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_num, query_length, d_model)
        :param key: (batch_num, key_length, d_model)
        :param value: (batch_num, key_length, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(
            query, key, value, mask=mask)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
