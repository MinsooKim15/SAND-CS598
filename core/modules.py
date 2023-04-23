import math

import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_prob=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor.
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DenseInterpolation(nn.Module):
    def __init__(self, seq_len: int, factor: int) -> None:
        """
        :param seq_len: sequence length
        :param factor: factor M
        """
        super(DenseInterpolation, self).__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1+m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2)


class ClassificationModule(nn.Module):
    def __init__(self, d_model: int, factor: int, num_class: int) -> None:
        super(ClassificationModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.num_class = num_class

        self.fc = nn.Linear(int(d_model * factor), num_class)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x


class RegressionModule(nn.Module):
    def __init__(self, d_model: int, factor: int, output_size: int) -> None:
        super(RegressionModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.output_size = output_size
        self.fc = nn.Linear(int(d_model * factor), output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))
        x = self.fc(x)
        return x
