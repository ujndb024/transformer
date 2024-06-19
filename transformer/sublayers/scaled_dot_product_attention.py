import torch.nn as nn
from torch import Tensor
from typing import Tuple

import math


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: float = 0.1) -> None:
        """
        Calculate `ScaledDotProductAttention`.
        It computes the attention scores between a query and a set of key-value pairs by taking the dot product between the query and the keys,
        followed by scaling and applying a softmax function.
            Parameters:
                `d_model`: int, required (default=512)
                    the embed dimension of the model.
                `num_heads`: int, required (default=8)
                    specifies how many parallel attention heads will be used.
                `dropout`: float, optional (default=0.1)
                    the dropout value.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Scaled Dot-Product Attention.

            Parameters:
                `query`: torch.Tensor; Tensor of shape [batch_size, num_heads, query_sequence_length, depth]
                    Tensor representing queries.
                `key`: torch.Tensor; Tensor of shape [batch_size, num_heads, key_sequence_length, depth]
                    Tensor representing keys.
                `value`: torch.Tensor; Tensor of shape [batch_size, num_heads, value_sequence_length, depth].
                    Tensor representing values.
                `mask`: torch.Tensor, optional; Tensor of shape [batch_size, 1, 1, key_sequence_length] or tensor of shape [batch_size, 1, key_sequence_length, key_sequence_length].
                    Tensor used for masking unmeaningful values during attention computation.

            Returns:
                `output, attention_weights`: Tuple[torch.Tensor, torch.Tensor];
                    Returns tuple where both tensor have the shape of [batch_size, num_heads, query_sequence_length, depth].
                    The output tensor representing the context vector obtained by weighted summing the values
                    based on the attention weights. It has the same shape as the input `query`.
        """
        depth = key.shape[-1]
        key_T = key.transpose(dim0=-2, dim1=-1)
        attention_scores = query.matmul(key_T) / math.sqrt(depth)

        if mask is not None:
            attention_scores = attention_scores.masked_fill_(
                mask=mask, value=float("-inf")
            )

        attention_weights = self.softmax(input=attention_scores)

        output = attention_weights.matmul(value)

        return output, attention_weights
