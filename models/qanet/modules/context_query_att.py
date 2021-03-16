import torch
from torch import nn as nn

from util import masked_softmax
from .initialized_conv import Initialized_Conv1d


class ContextQueryAttention(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.c_weight = nn.Parameter(torch.zeros(model_dim, 1))
        self.q_weight = nn.Parameter(torch.zeros(model_dim, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, model_dim))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))
        self.resizer = Initialized_Conv1d(model_dim * 4, model_dim)

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)

        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        s1 = masked_softmax(s, q_mask.view(batch_size, 1, q_len), dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask.view(batch_size, c_len, 1), dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return self.resizer(x)

    def get_similarity_matrix(self, c, q):
        batch_size, c_len, model_dim = c.shape
        _, q_len, _ = q.shape
        c, q = self.dropout(c), self.dropout(q)

        s0 = (
            torch
                .matmul(c, self.c_weight)
                .expand([-1, -1, q_len])
        )
        s1 = (
            torch
                .matmul(q, self.q_weight)
                .transpose(1, 2)
                .expand([-1, c_len, -1])
        )
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
