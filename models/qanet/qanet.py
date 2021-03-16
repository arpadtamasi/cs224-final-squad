# -*- coding: utf-8 -*-
"""
Main model architecture.
reference: https://github.com/andy840314/QANet-pytorch-
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from .modules.embedding import Embedding
from .modules.encoder import EncoderBlockConf, Encoder
from .modules.initialized_conv import Initialized_Conv1d


@dataclass
class QANetConf:
    freeze_char_embedding: bool = False
    dropout: float = 0.1
    char_dropout: float = 0.05
    model_dim: int = 128
    context_len: int = 401
    question_len: int = 51
    pad: int = 0
    use_mask_in_ee: bool = False
    use_mask_in_cq: bool = False
    use_mask_in_me: bool = False
    use_mask_in_ou: bool = False

    embedding: EncoderBlockConf = EncoderBlockConf(
        kernel_size=7, layer_dropout=0.9, dropout=0.1,
        num_heads=8, num_convs=4, num_blocks=1, performer=None
    )
    modeling: EncoderBlockConf = EncoderBlockConf(
        kernel_size=5, layer_dropout=0.9, dropout=0.1,
        num_heads=8, num_convs=2, num_blocks=7, performer=None
    )


class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, config: QANetConf):
        super(QANet, self).__init__()

        self.config = config
        self.embedder = Embedding(
            word_vectors, char_vectors, config.model_dim,
            word_dropout=config.dropout, char_dropout=config.char_dropout,
            freeze_char_embedding=config.freeze_char_embedding
        )

        self.embedding_encoder = Encoder(config.embedding, config.model_dim)

        self.context_query_attention = ContextQueryAttention(
            model_dim=config.model_dim,
            dropout=config.dropout
        )

        from models.qanet.cq_att import CQAttention
        self.cq_att = CQAttention(
            d_model=config.model_dim,
            dropout=config.dropout
        )

        self.model_encoder = Encoder(config.modeling, config.model_dim)

        self.output = Pointer(config.model_dim)


    def forward(self,
                context_word_ids, context_char_ids,
                query_word_ids, query_char_ids):
        context_mask = torch.zeros_like(context_word_ids) != context_word_ids
        query_mask = torch.zeros_like(query_word_ids) != query_word_ids

        cme = context_mask if self.config.use_mask_in_ee else None
        qme = query_mask if self.config.use_mask_in_ee else None
        cmq, qmq = (context_mask, query_mask) if self.config.use_mask_in_cq else (None, None)
        cmm = context_mask if self.config.use_mask_in_me else None
        cmo = context_mask if self.config.use_mask_in_ou else None

        # context embedding and encoding
        context_emb = self.embedder(context_word_ids, context_char_ids)
        context_enc = self.embedding_encoder(context_emb, mask=cme)

        # query embedding and encoding
        query_emb = self.embedder(query_word_ids, query_char_ids)
        query_enc = self.embedding_encoder(query_emb, mask=qme)

        # context-query attention
        #context_query_enc = self.context_query_attention(context_enc, query_enc, Cmask=cmq, Qmask=qmq)
        context_query_enc = self.cq_att(context_enc, query_enc, Cmask=cmq, Qmask=qmq)

        # model encoding
        from .modules.functional import recurse
        m0, m1, m2 = recurse(func=lambda m: self.model_encoder(m, mask=cmm), initial=context_query_enc, max=3)

        # output layer
        log_p1, log_p2 = self.output(m0, m1, m2, mask=cmo)

        return log_p1, log_p2



class ContextQueryAttention(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        w4C = torch.empty(model_dim, 1)
        w4Q = torch.empty(model_dim, 1)
        w4mlu = torch.empty(1, 1, model_dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = nn.Dropout(dropout)
        self.W0 = nn.Linear(model_dim * 3, 1, bias=False)

        self.resizer = Initialized_Conv1d(model_dim * 4, model_dim)

    def forward(self, context, question, Cmask=None, Qmask=None):
        c = context
        q = question
        S = self.trilinear(c, q)
        S_ = torch.softmax(S, 1)
        A = torch.bmm(S_, q)

        S__ = torch.softmax(S, 2)
        s__T = S__.permute(0, 2, 1)
        B = torch.bmm(torch.bmm(S_, s__T), c)

        out = torch.cat([c, A, c * A, c * B], dim=-1)

        return self.resizer(out)

    def trilinear(self, context, question):
        batch_size, context_len, model_dim = context.shape
        _, question_len, _ = question.shape
        similarity_matrix_shape = torch.zeros(batch_size, context_len, question_len, model_dim)
        c = context.unsqueeze(2).expand_as(similarity_matrix_shape)
        q = question.unsqueeze(1).expand_as(similarity_matrix_shape)
        f_cq = torch.cat([c, q, c * q], dim=-1)
        return self.W0(f_cq).squeeze(-1)



class Pointer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model * 2, 1, bias=False)
        self.w2 = nn.Linear(d_model * 2, 1, bias=False)

    def forward(self, m0, m1, m2, mask=None):
        x_start = torch.cat([m0, m1], dim=-1)
        logits_start = self.w1(x_start).squeeze(-1)
        log_p_start = torch.log_softmax(logits_start, dim=-1).squeeze(-1)

        x_end = torch.cat([m0, m2], dim=-1)
        logits_end = self.w2(x_end).squeeze(-1)
        log_p_end = torch.log_softmax(logits_end, dim=-1).squeeze(-1)

        return log_p_start, log_p_end
