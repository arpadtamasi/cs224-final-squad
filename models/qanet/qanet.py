from dataclasses import dataclass

import torch
import torch.nn as nn

from .modules.context_query_att import ContextQueryAttention
from .modules.embedding import Embedding
from .modules.encoder import EncoderBlockConf, Encoder
from .modules.qanet_output import QANetOutput
from .modules.util import recurse


@dataclass
class QANetConf:
    freeze_char_embedding: bool = False
    dropout: float = 0.1
    char_dropout: float = 0.05
    model_dim: int = 128

    input_encoder: EncoderBlockConf = EncoderBlockConf(
        kernel_size=7, layer_dropout=0.9, dropout=0.1,
        num_heads=8, num_convs=4, num_blocks=1, performer=None
    )
    model_encoder: EncoderBlockConf = EncoderBlockConf(
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

        self.input_encoder = Encoder(config.input_encoder, config.model_dim)

        self.context_query_attention = ContextQueryAttention(
            model_dim=config.model_dim,
            dropout=config.dropout
        )

        self.model_encoder = Encoder(config.model_encoder, config.model_dim)

        self.output = QANetOutput(config.model_dim)

    def forward(self,
                context_word_ids, context_char_ids,
                query_word_ids, query_char_ids):
        # context embedding and encoding
        context = self.input_encoder(
            self.embedder(context_word_ids, context_char_ids)
        )

        # query embedding and encoding
        query = self.input_encoder(
            self.embedder(query_word_ids, query_char_ids)
        )

        # context-query attention
        model = self.context_query_attention(
            context, query,
            c_mask=(torch.zeros_like(context_word_ids) != context_word_ids),
            q_mask=(torch.zeros_like(query_word_ids) != query_word_ids)
        )

        # model encoding
        m0, m1, m2 = recurse(func=self.model_encoder, initial=model, max=3)

        # output layer
        log_p1, log_p2 = self.output(m0, m1, m2)

        return log_p1, log_p2
