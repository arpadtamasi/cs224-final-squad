# -*- coding: utf-8 -*-
"""
Main model architecture.
reference: https://github.com/andy840314/QANet-pytorch-
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.cnn import DepthwiseSeparableConv
from .modules.embedding import Embedding
from .modules.functional import mask_logits
from .modules.initialized_conv1d import Initialized_Conv1d
from .modules.positional_encoding import PositionalEncoding
from .modules.self_attention import SelfAttention


@dataclass
class EncoderBlockConf:
    kernel_size: int
    layer_dropout: float
    dropout: float
    num_heads: int
    num_convs: int
    num_blocks: int


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
    def __init__(self, word_mat, char_mat, config: QANetConf):  # !!! notice: set it to be a config parameter later.
        super(QANet, self).__init__()

        self.config = config
        self.emb = Embedding(
            word_mat, char_mat, config.model_dim,
            dropout_w=config.dropout, dropout_c=config.char_dropout,
            freeze_char_embedding=config.freeze_char_embedding
        )

        self.emb_enc_blks = nn.ModuleList([
            EncoderBlock(
                conv_num=config.embedding.num_convs,
                d_model=config.model_dim,
                num_head=config.embedding.num_heads,
                k=config.embedding.kernel_size,
                block_index=i, num_blocks=config.embedding.num_blocks,
                layer_dropout = config.embedding.layer_dropout,
                dropout=config.embedding.dropout
            )
            for i in range(config.embedding.num_blocks)
        ])

        self.cq_att = CQAttention(d_model=config.model_dim)
        self.cq_resizer = Initialized_Conv1d(config.model_dim * 4, config.model_dim)
        self.model_enc_blks = nn.ModuleList([
            EncoderBlock(
                conv_num=config.modeling.num_convs,
                d_model=config.model_dim,
                num_head=config.modeling.num_heads,
                k=config.modeling.kernel_size,
                block_index=i, num_blocks=config.modeling.num_blocks,
                layer_dropout = config.modeling.layer_dropout,
                dropout=config.modeling.dropout
            )
            for i in range(config.modeling.num_blocks)
        ])
        self.out = Pointer(config.model_dim)
        self.PAD = config.pad
        self.Lc = config.context_len
        self.Lq = config.question_len
        self.dropout = config.dropout

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.ones_like(Cwid) * self.PAD != Cwid).float()
        maskQ = (torch.ones_like(Qwid) * self.PAD != Qwid).float()
        C, Q = self.emb(Cwid, Ccid), self.emb(Qwid, Qcid)

        Ce, Qe = C, Q
        for i, blk in enumerate(self.emb_enc_blks):
            Ce = blk(Ce, maskC)
            Qe = blk(Qe, maskQ)

        X = self.cq_att(Ce, Qe, maskC, maskQ)

        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC)
        M3 = M0
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
        l = 1
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
        L = self.total_blocks * (self.conv_num + 1) + 1
        prob = ((l / L) * (1 - self.layer_dropout_prob))
        if self.training == True:
            survive = torch.empty(1).uniform_(0, 1) < prob
            if survive:
                return residual
            else:
                return F.dropout(inputs, prob, training=self.training) + residual
        else:
            return inputs + residual


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
