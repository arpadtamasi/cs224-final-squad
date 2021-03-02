import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from .depthwise_separable_conv import DepthwiseSeparableConv
from .functions import mask_logits
from .pos_encoder import PosEncoder


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, n_head):
        super(SelfAttention, self).__init__()
        Wo = torch.empty(d_model, d_k * n_head)
        Wqs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wks = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wvs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        nn.init.kaiming_uniform_(Wo)
        for i in range(n_head):
            nn.init.xavier_uniform_(Wqs[i])
            nn.init.xavier_uniform_(Wks[i])
            nn.init.xavier_uniform_(Wvs[i])
        self.Wo = nn.Parameter(Wo)
        self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
        self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
        self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])
        self.d_k = d_k
        self.n_head = n_head

    def forward(self, x, mask):
        WQs, WKs, WVs = [], [], []
        sqrt_d_k_inv = 1 / math.sqrt(self.d_k)
        x = x.transpose(1, 2)
        hmask = mask.unsqueeze(1)
        vmask = mask.unsqueeze(2)
        for i in range(self.n_head):
            WQs.append(torch.matmul(x, self.Wqs[i]))
            WKs.append(torch.matmul(x, self.Wks[i]))
            WVs.append(torch.matmul(x, self.Wvs[i]))
        heads = []
        for i in range(self.n_head):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
            out = torch.mul(out, sqrt_d_k_inv)
            # not sure... I think `dim` should be 2 since it weighted each column of `WVs[i]`
            out = mask_logits(out, hmask)
            out = F.softmax(out, dim=2) * vmask
            headi = torch.bmm(out, WVs[i])
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        d_k = d_model // n_head
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.a = 1 / math.sqrt(d_k)
        self.d_k = d_k
        self.n_head = n_head
        self.d_model = d_model

    def forward(self, x, mask):
        bs, _, l_x = x.size()
        x = x.transpose(1, 2)
        k = self.k_linear(x).view(bs, l_x, self.n_head, self.d_k)
        q = self.q_linear(x).view(bs, l_x, self.n_head, self.d_k)
        v = self.v_linear(x).view(bs, l_x, self.n_head, self.d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs * self.n_head, l_x, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs * self.n_head, l_x, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs * self.n_head, l_x, self.d_k)
        mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(self.n_head, 1, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.a
        attn = mask_logits(attn, mask)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        out = out.view(self.n_head, bs, l_x, self.d_k).permute(1, 2, 0, 3).contiguous().view(bs, l_x, self.d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int, d_model, n_head, dropout):
        super(EncoderBlock, self).__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = MultiHeadAttention(d_model, n_head, dropout)
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = PosEncoder(length, d_model)
        # self.norm = nn.LayerNorm([d_model, length])
        self.normb = nn.LayerNorm([d_model, length])
        self.norms = nn.ModuleList([nn.LayerNorm([d_model, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([d_model, length])
        self.L = conv_num
        self.dropout = dropout

    def forward(self, x, mask):
        out = self.pos(x)
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        # print("Before attention: {}".format(out.size()))
        out = self.self_att(out, mask)
        # print("After attention: {}".format(out.size()))
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out
