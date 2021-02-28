"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn

import models.bidaf.modules.bidaf_attention
import models.bidaf.modules.bidaf_output
import models.bidaf.modules.embedding
import models.bidaf.modules.rnn_encoder


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
        self.emb = models.bidaf.modules.embedding.Embedding(word_vectors=word_vectors,
                                                            hidden_size=hidden_size,
                                                            drop_prob=drop_prob)

        self.enc = models.bidaf.modules.rnn_encoder.RNNEncoder(input_size=hidden_size,
                                                               hidden_size=hidden_size,
                                                               num_layers=1,
                                                               drop_prob=drop_prob)

        self.att = models.bidaf.modules.bidaf_attention.BiDAFAttention(hidden_size=2 * hidden_size,
                                                                       drop_prob=drop_prob)

        self.mod = models.bidaf.modules.rnn_encoder.RNNEncoder(input_size=8 * hidden_size,
                                                               hidden_size=hidden_size,
                                                               num_layers=2,
                                                               drop_prob=drop_prob)

        self.out = models.bidaf.modules.bidaf_output.BiDAFOutput(hidden_size=hidden_size,
                                                                 drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out