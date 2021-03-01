"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn

from .modules.bidaf_attention import BiDAFAttention
from .modules.bidaf_output import BiDAFOutput
from .modules.embedding import Embedding
from .modules.rnn_encoder import RNNEncoder


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = Embedding(
            word_vectors=word_vectors,
            hidden_size=hidden_size,
            drop_prob=drop_prob
        )

        self.enc = RNNEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            drop_prob=drop_prob
        )

        self.att = BiDAFAttention(
            hidden_size=2 * hidden_size,
            drop_prob=drop_prob
        )

        self.mod = RNNEncoder(
            input_size=8 * hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            drop_prob=drop_prob
        )

        self.out = BiDAFOutput(
            hidden_size=hidden_size,
            drop_prob=drop_prob
        )

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs # (batch_size, c_len)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs # (batch_size, q_len)

        c_len = c_mask.sum(-1) # (batch_size)
        q_len = q_mask.sum(-1) # (batch_size)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(
            c_enc, q_enc,
            c_mask, q_mask
        )  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        log_p1, log_p2 = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        spans = list(zip(
            torch.argmax(log_p1, -1).tolist(),
            torch.argmax(log_p2, -1).tolist()
        ))
        return log_p1, log_p2
