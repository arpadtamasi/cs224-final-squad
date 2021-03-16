import torch
from torch import nn as nn


class QANetOutput(nn.Module):
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
