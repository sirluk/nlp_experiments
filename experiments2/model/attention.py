from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Attention(nn.Module):
    def __init__(self, query_dim: int):
        super().__init__()

        self.query = nn.Parameter(torch.randn(size=(1, 1, query_dim)))

        self.scale = 1.0 / np.sqrt(query_dim)

    def forward(self, values: Tensor, padding_mask: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = values.shape[0]

        # [batch_size, 1, query_dim]
        queries = self.query.repeat(batch_size, 1, 1)

        weights = self._get_attention_weights(queries, values) * self.scale
        weights = weights.masked_fill(padding_mask.unsqueeze(1), float('-inf'))

        weights = F.softmax(weights, dim=-1)

        # [batch_size, value_dim]
        return (weights @ values).squeeze(1), weights

    def _get_attention_weights(self, queries: Tensor, values: Tensor) -> Tensor:
        raise NotImplementedError()


class DotProductAttention(Attention):
    def __init__(self, query_dim: int):
        super().__init__(query_dim)

    def _get_attention_weights(self, queries: Tensor, values: Tensor) -> Tensor:
        # [batch_size, key_dim, sequence_length]
        keys = values.permute(0, 2, 1)

        # [batch_size, 1, sequence_length]
        return queries @ keys


class MultiplicativeAttention(Attention):
    def __init__(self, query_dim: int):
        super().__init__(query_dim)

        self.W = nn.Parameter(torch.randn(query_dim, query_dim))

    def _get_attention_weights(self, queries: Tensor, values: Tensor) -> Tensor:
        # [batch_size, key_dim, sequence_length]de
        keys = values.permute(0, 2, 1)

        # [batch_size, 1, sequence_length]
        return queries @ self.W @ keys


class AdditiveAttention(Attention):
    def __init__(self, query_dim: int):
        super().__init__(query_dim)

        self.W = nn.Linear(2 * query_dim, query_dim, bias=False)
        self.u = nn.Parameter(torch.randn(1, query_dim, 1))

    def _get_attention_weights(self, queries: Tensor, values: Tensor) -> Tensor:
        batch_size, sequence_length, _ = values.shape

        # [batch_size, sequence_length, query_dim]
        queries = queries.repeat(1, sequence_length, 1)
        return (torch.tanh(self.W(torch.cat((queries, values), dim=-1))) @ self.u).permute(0, 2, 1)
