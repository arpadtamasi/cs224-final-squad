import torch
import torch.nn as nn

from models.qanet.modules.embedding import Embedding
from .modules import attention
from .modules import conv
from .modules import encoder
from .modules import layer


class QANet(nn.Module):
    def __init__(
            self,
            word_vectors, char_vectors,
            aligned_query_embedding=True, freeze_char_embedding=False,
            model_dim=128,
            kernel_size_in_embedding=7,
            num_head_in_embedding=8,
            num_conv_block_in_embedding=2,
            num_embedding_encoder_block=1,
            kernel_size_in_modeling=5,
            num_head_in_modeling=8,
            num_conv_block_in_modeling=2,
            num_modeling_encoder_block=7,
            dropout=0.1,
            layer_dropout=0.9,
            char_dropout=0.05
    ):
        super(QANet, self).__init__()

        word_dropout = dropout
        char_dropout = char_dropout
        embed_encoder_dropout = dropout
        embed_encoder_layer_dropout = layer_dropout
        model_encoder_dropout = 0
        model_encoder_layer_dropout = layer_dropout
        cq_dropout = dropout

        if aligned_query_embedding:
            emb = Embedding(word_vectors, char_vectors, freeze_char_embedding, word_dropout, char_dropout)
            embed_pointwise_conv = conv.PointwiseConv(emb.d_embed, model_dim)

            self.c_emb = emb
            self.q_emb = emb

            self.c_conv = embed_pointwise_conv
            self.q_conv = embed_pointwise_conv

        else:
            self.c_emb = Embedding(word_vectors, char_vectors, freeze_char_embedding, word_dropout, char_dropout)
            self.q_emb = Embedding(word_vectors, char_vectors, freeze_char_embedding, word_dropout, char_dropout)

            self.c_conv = conv.PointwiseConv(self.c_emb, model_dim)
            self.q_conv = conv.PointwiseConv(self.q_emb, model_dim)
        self.embed_encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    model_dim=model_dim,
                    kernel_size=kernel_size_in_embedding,
                    num_head=num_head_in_embedding,
                    num_conv_block=num_conv_block_in_modeling,
                    dropout=embed_encoder_dropout,
                    layer_dropout=embed_encoder_layer_dropout,
                )
                for _ in range(num_embedding_encoder_block)
            ]
        )

        self.co_attention = attention.CoAttention(model_dim)
        self.pointwise_conv = conv.PointwiseConv(model_dim * 4, model_dim)

        self.model_encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    model_dim=model_dim,
                    kernel_size=kernel_size_in_modeling,
                    num_head=num_head_in_modeling,
                    num_conv_block=num_conv_block_in_modeling,
                    dropout=model_encoder_dropout,
                    layer_dropout=model_encoder_layer_dropout,
                )
                for _ in range(num_modeling_encoder_block)
            ]
        )

        self.pointer = Pointer(model_dim)

        self.cq_dropout = nn.Dropout(p=cq_dropout)

    # noinspection PyUnresolvedReferences
    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()  # batch_size x para_limit
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()  # batch_size x ques_limit

        c_emb = self.c_emb(cw_idxs, cc_idxs)  # batch_size x para_limit x d_embed
        q_emb = self.q_emb(qw_idxs, qc_idxs)  # batch_size x ques_limit x d_embed

        c_conv = self.c_conv(c_emb.permute(0, 2, 1))  # batch_size x d_model x para_limit
        q_conv = self.q_conv(q_emb.permute(0, 2, 1))  # batch_size x d_model x ques_limit

        context = c_conv
        query = q_conv
        for encoder_block in self.embed_encoder_blocks:
            context = encoder_block(context)
            query = encoder_block(query)

        # 3. Context-Query Attention Layer
        context_query_attention = self.co_attention(context, query, c_mask, q_mask)

        # Projection (memory issue)
        context_query_attention = self.pointwise_conv(context_query_attention)
        context_query_attention = self.cq_dropout(context_query_attention)

        # 4. Model Encoder Layer
        model_encoder_block_inputs = context_query_attention

        # Stacked Model Encoder Block
        stacked_model_encoder_blocks = []
        for i in range(3):
            for _, model_encoder_block in enumerate(self.model_encoder_blocks):
                output = model_encoder_block(model_encoder_block_inputs, c_mask)
                model_encoder_block_inputs = output

            stacked_model_encoder_blocks.append(output)

        return self.pointer(*stacked_model_encoder_blocks, c_mask)


class EncoderBlock(nn.Module):
    """
        Encoder Block

        []: residual
        position_encoding -> [convolution-layer] x # -> [self-attention-layer] -> [feed-forward-layer]

        - convolution-layer: depthwise separable convolutions
        - self-attention-layer: multi-head attention
        - feed-forward-layer: pointwise convolution

        * Args:
            model_dim: the number of model dimension
            num_heads: the number of head in multi-head attention
            kernel_size: convolution kernel size
            num_conv_block: the number of convolution block
            dropout: the dropout probability
            layer_dropout: the layer dropout probability
                (cf. Deep Networks with Stochastic Depth(https://arxiv.org/abs/1603.09382) )
    """

    def __init__(
            self,
            model_dim=128,
            num_head=8,
            kernel_size=5,
            num_conv_block=4,
            dropout=0.1,
            layer_dropout=0.9,
    ):
        super(EncoderBlock, self).__init__()

        self.position_encoding = encoder.PositionalEncoding(model_dim)
        self.dropout = nn.Dropout(dropout)

        self.num_conv_block = num_conv_block
        self.conv_blocks = nn.ModuleList(
            [conv.DepSepConv(model_dim, model_dim, kernel_size) for _ in range(num_conv_block)]
        )

        self.self_attention = attention.MultiHeadAttention(
            num_head=num_head, model_dim=model_dim, dropout=dropout
        )
        self.feedforward_layer = layer.PositionwiseFeedForward(
            model_dim, model_dim * 4, dropout=dropout
        )

        # survival probability for stochastic depth
        if layer_dropout < 1.0:
            L = (num_conv_block) + 2 - 1
            layer_dropout_prob = round(1 - (1 / L) * (1 - layer_dropout), 3)
            self.residuals = nn.ModuleList(
                layer.ResidualConnection(
                    model_dim, layer_dropout=layer_dropout_prob, layernorm=True
                )
                for l in range(num_conv_block + 2)
            )
        else:
            self.residuals = nn.ModuleList(
                layer.ResidualConnection(model_dim, layernorm=True)
                for l in range(num_conv_block + 2)
            )

    def forward(self, x, mask=None):
        # Positional Encoding
        x = self.position_encoding(x)

        # Convolution Block (LayerNorm -> Conv)
        for i, conv_block in enumerate(self.conv_blocks):
            x = self.residuals[i](x, sub_layer_fn=conv_block)
            x = self.dropout(x)

        # LayerNorm -> Self-attention
        self_attention = lambda x: self.self_attention(q=x, k=x, v=x, mask=mask)
        x = self.residuals[self.num_conv_block](x, sub_layer_fn=self_attention)
        x = self.dropout(x)

        # LayerNorm -> Feedforward layer
        x = self.residuals[self.num_conv_block + 1](x, sub_layer_fn=self.feedforward_layer)
        x = self.dropout(x)
        return x


class Pointer(nn.Module):
    def __init__(self, model_dim):
        super().__init__()

        self.span_start_linear = nn.Linear(model_dim * 2, 1, bias=False)
        self.span_end_linear = nn.Linear(model_dim * 2, 1, bias=False)

    def forward(self, m0, m1, m2, mask):
        x1 = torch.cat([m0, m1], dim=-1)
        x2 = torch.cat([m0, m2], dim=-1)

        from util import masked_softmax
        log_p1 = masked_softmax(self.span_start_linear(x1).squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(self.span_end_linear(x2).squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
