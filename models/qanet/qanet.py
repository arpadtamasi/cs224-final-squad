import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.attention import EncoderBlock
from .modules.depthwise_separable_conv import DepthwiseSeparableConv
from .modules.embedding import Embedding
from .modules.functions import mask_logits


@dataclass
class EncoderBlockConf:
    kernel_size: int
    layer_dropout: float
    num_heads: int
    num_convs: int
    num_blocks: int

@dataclass
class QANetConf:
    aligned_query_embedding: bool = True
    freeze_char_embedding: bool = False
    dropout: float = 0.1
    char_dropout: float = 0.05
    model_dim: int = 128

    context_len: int = 401
    question_len: int = 51

    embedding: EncoderBlockConf = EncoderBlockConf(
        kernel_size=7, layer_dropout=0.9,
        num_heads=8, num_convs=4, num_blocks=1
    )
    modeling: EncoderBlockConf = EncoderBlockConf(
        kernel_size=5, layer_dropout=0.9,
        num_heads=8, num_convs=2, num_blocks=7
    )

class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, config: QANetConf):
        super(QANet, self).__init__()

        self.emb = Embedding(word_vectors, char_vectors, config.freeze_char_embedding, config.dropout, config.char_dropout)
        d_embed = self.emb.d_embed

        self.c_conv = DepthwiseSeparableConv(d_embed, config.model_dim, 5)
        self.q_conv = DepthwiseSeparableConv(d_embed, config.model_dim, 5)

        self.c_emb_enc = EncoderBlock(
            conv_num=config.embedding.num_convs,
            ch_num=config.model_dim,
            k=config.embedding.kernel_size,
            length=config.context_len,
            d_model=config.model_dim,
            n_head=config.embedding.num_heads,
            dropout=config.dropout
        )
        self.q_emb_enc = EncoderBlock(
            conv_num=config.embedding.num_convs,
            ch_num=config.model_dim,
            k=config.embedding.kernel_size,
            length=config.question_len,
            d_model=config.model_dim,
            n_head=config.embedding.num_heads,
            dropout=config.dropout
        )

        self.cq_att = CQAttention(config.model_dim, config.dropout)
        self.cq_resizer = DepthwiseSeparableConv(config.model_dim * 4, config.model_dim, 5)

        enc_blk = EncoderBlock(
            conv_num=config.modeling.num_convs,
            ch_num=config.model_dim,
            k=config.modeling.kernel_size,
            length=config.context_len,
            d_model=config.model_dim,
            n_head=config.modeling.num_heads,
            dropout=config.dropout
        )
        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)

        self.linear_start = nn.Linear(config.model_dim * 2, 1)
        self.linear_end = nn.Linear(config.model_dim * 2, 1)

        self.dropout = nn.Dropout(config.dropout)

    # noinspection PyUnresolvedReferences
    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()  # batch_size x para_limit
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()  # batch_size x ques_limit

        c_emb = self.emb(cw_idxs, cc_idxs)  # batch_size x para_limit x d_embed
        q_emb = self.emb(qw_idxs, qc_idxs)  # batch_size x ques_limit x d_embed

        c_conv = self.c_conv(c_emb)  # batch_size x d_model x para_limit
        q_conv = self.q_conv(q_emb)  # batch_size x d_model x ques_limit

        c_enc = self.c_emb_enc(c_conv, c_mask)  # batch_size x d_model x para_limit
        q_enc = self.q_emb_enc(q_conv, q_mask)  # batch_size x d_model x ques_limit

        x = self.cq_att(c_enc, q_enc, c_mask, q_mask)  # batch_size x (d_model * 4) x para_limit

        m1 = self.cq_resizer(x)  # batch_size x d_model x para_limit
        for enc in self.model_enc_blks:
            m1 = enc(m1, c_mask)
        m2 = m1
        for enc in self.model_enc_blks:
            m2 = enc(m2, c_mask)
        m3 = m2
        for enc in self.model_enc_blks:
            m3 = enc(m3, c_mask)

        x1 = torch.cat([m1, m2], dim = 1)
        x2 = torch.cat([m1, m3], dim = 1)

        logits_start = self.linear_start(x1.permute(0, 2, 1))  # batch_size x para_limit
        logits_end = self.linear_end(x2.permute(0, 2, 1))  # batch_size x para_limit

        from util import masked_softmax
        log_p1 = masked_softmax(logits_start.squeeze(), c_mask, log_softmax=True)
        log_p2 = masked_softmax(logits_end.squeeze(), c_mask, log_softmax=True)

        return log_p1, log_p2


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super(CQAttention, self).__init__()
        w = torch.empty(d_model * 3)
        lim = 1 / d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)
        self.dropout = dropout

    def forward(self, C, Q, cmask, qmask):
        ss = []
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)

        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        S = torch.cat([Ct, Qt, CQ], dim=3)
        S = torch.matmul(S, self.w)
        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out.transpose(1, 2)


