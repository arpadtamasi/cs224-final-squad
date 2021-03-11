import torch
from torch import nn as nn
from torch.nn import functional as F

from .depthwise_separable_conv import DepthwiseSeparableConv
from .highway import Highway


class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, freeze_char_embedding, dropout, dropout_char):
        super(Embedding, self).__init__()

        d_word = word_vectors.shape[-1]
        d_char = char_vectors.shape[-1]

        self.d_embed = d_word + d_char
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors), freeze=freeze_char_embedding)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_vectors))

        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2)
        self.high = Highway(2, d_word + d_char)
        self.dropout = nn.Dropout(dropout)
        self.char_dropout = nn.Dropout(dropout_char)

    def forward(self, w_idxs, c_idxs):
        wd_emb, ch_emb = self.word_emb(w_idxs), self.char_emb(c_idxs)  # batch_size x len(w_idxs) x w_embed, batch_size x len(c_idxs) x char_limit x c_embed

        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = self.char_dropout(ch_emb)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=-1)
        wd_emb = self.dropout(wd_emb)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.high(emb)
        return emb
