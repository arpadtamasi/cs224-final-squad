import torch
from torch import nn as nn
from torch.nn import functional as F

from .highway import Highway
from .initialized_conv1d import Initialized_Conv1d


class Embedding(nn.Module):
    def __init__(self, word_mat, char_mat, d_model,
                 dropout_w=0.1, dropout_c=0.05, freeze_char_embedding=False):
        super().__init__()

        self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=freeze_char_embedding)
        self.word_emb = nn.Embedding.from_pretrained(word_mat)
        wemb_dim = word_mat.shape[1]
        cemb_dim = char_mat.shape[1]

        self.conv2d = nn.Conv2d(cemb_dim, d_model, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(wemb_dim + d_model, d_model, bias=False)
        self.high = Highway(dropout_w, 2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, w_ids, c_ids):
        wd_emb = self.word_emb(w_ids)
        ch_emb = self.char_emb(c_ids)

        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)

        return emb
