import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.depthwise_separable_conv import DepthwiseSeparableConv
from .modules.embedding import Embedding
from .modules.functions import mask_logits
from .modules.attention import EncoderBlock

class QANet(nn.Module):
    def __init__(self, word_mat, char_mat, d_word, d_char, d_model, n_head, len_c, len_q, dropout, dropout_char, freeze_char_embedding):
        super(QANet, self).__init__()
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=freeze_char_embedding)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.emb = Embedding(d_word, d_char, dropout, dropout_char)
        self.context_conv = DepthwiseSeparableConv(d_word + d_char, d_model, 5)
        self.question_conv = DepthwiseSeparableConv(d_word + d_char, d_model, 5)
        self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_c, d_model=d_model, n_head=n_head, dropout=dropout)
        self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_q, d_model=d_model, n_head=n_head, dropout=dropout)
        self.cq_att = CQAttention(d_model, dropout)
        self.cq_resizer = DepthwiseSeparableConv(d_model * 4, d_model, 5)
        enc_blk = EncoderBlock(conv_num=2, ch_num=d_model, k=5, length=len_c, d_model=d_model, n_head=n_head, dropout=dropout)
        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)
        self.out = Pointer(d_model)

    # noinspection PyUnresolvedReferences
    def forward(self, Cwid, Ccid, Qwid, Qcid):
        # Cwid, Ccid: batch_size x para_limit, batch_size x para_limit x char_limit
        # Qwid, Qcid: batch_size x ques_limit, batch_size x ques_limit x char_limit
        cmask = (torch.zeros_like(Cwid) == Cwid).float()  # batch_size x para_limit
        qmask = (torch.zeros_like(Qwid) == Qwid).float()  # batch_size x ques_limit
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)  # batch_size x para_limit x word_embed_size, batch_size x para_limit x char_limit x char_embed_size
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)  # batch_size x ques_limit x word_embed_size, batch_size x ques_limit x char_limit x char_embed_size
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)  # batch_size x (word_embed_size + char_embed_size) x para_limit, batch_size x (word_embed_size + char_embed_size) x ques_limit
        C = self.context_conv(C)  # batch_size x d_model x para_limit
        Q = self.question_conv(Q)  # batch_size x d_model x ques_limit
        Ce = self.c_emb_enc(C, cmask)  # batch_size x d_model x para_limit
        Qe = self.q_emb_enc(Q, qmask)  # batch_size x d_model x ques_limit

        X = self.cq_att(Ce, Qe, cmask, qmask)  # batch_size x (d_model * 4) x para_limit
        M1 = self.cq_resizer(X)  # batch_size x d_model x para_limit
        for enc in self.model_enc_blks:
            M1 = enc(M1, cmask)
        M2 = M1
        for enc in self.model_enc_blks:
            M2 = enc(M2, cmask)
        M3 = M2
        for enc in self.model_enc_blks:
            M3 = enc(M3, cmask)
        p1, p2 = self.out(M1, M2, M3, cmask)  # batch_size x para_limit, batch_size x para_limit,
        return p1, p2


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


class Pointer(nn.Module):
    def __init__(self, d_model):
        super(Pointer, self).__init__()
        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        lim = 3 / (2 * d_model)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = torch.matmul(self.w1, X1)
        Y2 = torch.matmul(self.w2, X2)
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2


