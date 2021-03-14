import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_length=2000):
        super(PositionalEncoding, self).__init__()
        signal = self._get_timing_signal(max_length, embed_dim)

        self.register_buffer("position_encoding", signal)

    def forward(self, x):
        x = x + self.position_encoding[:, :, :x.size(-1)]
        return x

    def _get_timing_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale) / (float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * np.exp(
            np.arange(num_timescales).astype(np.float) * -log_timescale_increment
        )
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0])
        signal = signal.reshape([1, length, channels])

        return torch.from_numpy(signal).type(torch.FloatTensor).transpose(1, 2)
