# -*- coding: utf-8 -*-
"""
Main model architecture.
reference: https://github.com/andy840314/QANet-pytorch-
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.embedding import Embedding
from .modules.encoder import EncoderBlockConf, Encoder
from .modules.functional import mask_logits
from .modules.initialized_conv1d import Initialized_Conv1d


@dataclass
class QANetConf:
    freeze_char_embedding: bool = False
    dropout: float = 0.1
    char_dropout: float = 0.05
    model_dim: int = 128
    context_len: int = 401
    question_len: int = 51
    pad: int = 0

    embedding: EncoderBlockConf = EncoderBlockConf(
        kernel_size=7, layer_dropout=0.9, dropout=0.1,
        num_heads=8, num_convs=4, num_blocks=1
    )
    modeling: EncoderBlockConf = EncoderBlockConf(
        kernel_size=5, layer_dropout=0.9, dropout=0.1,
        num_heads=8, num_convs=2, num_blocks=7
    )


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat, config: QANetConf):
        super(QANet, self).__init__()

        self.config = config
        self.emb = Embedding(
            word_mat, char_mat, config.model_dim,
            dropout_w=config.dropout, dropout_c=config.char_dropout,
            freeze_char_embedding=config.freeze_char_embedding
        )

        self.embedding_encoder = Encoder(config.embedding, config.model_dim)

        self.cq_att = CQAttention(d_model=config.model_dim)
        self.cq_resizer = Initialized_Conv1d(config.model_dim * 4, config.model_dim)

        self.model_encoder = Encoder(config.modeling, config.model_dim)

        self.out = Pointer(config.model_dim)
        self.PAD = config.pad
        self.Lc = config.context_len
        self.Lq = config.question_len
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.ones_like(Cwid) * self.PAD != Cwid).float()
        maskQ = (torch.ones_like(Qwid) * self.PAD != Qwid).float()
        C, Q = self.emb(Cwid, Ccid), self.emb(Qwid, Qcid)

        Ce = self.embedding_encoder(C, maskC)
        Qe = self.embedding_encoder(Q, maskQ)

        X = self.cq_att(Ce, Qe, maskC, maskQ)
        X = self.cq_resizer(X)
        X = self.dropout(X)

        M1 = self.model_encoder(X, maskC)
        M2 = self.model_encoder(M1, maskC)
        M3 = self.model_encoder(M2, maskC)

        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model * 2, 1)
        self.w2 = Initialized_Conv1d(d_model * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        L1 = self.w1(X1)
        L2 = self.w2(X2)

        Y1 = mask_logits(L1.squeeze(), mask)
        Y2 = mask_logits(L2.squeeze(), mask)

        from util import masked_softmax
        log_p1 = masked_softmax(Y1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(Y2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
