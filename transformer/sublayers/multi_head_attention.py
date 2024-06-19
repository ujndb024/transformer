import torch
import torch.nn as nn
from torch import Tensor

from transformer.sublayers.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        device: torch.device,
        d_model: int = 512,
        num_heads: int = 8,
    ) -> None:
        """
        `MultiHeadAttention` is applied multiple times in parallel, each with its own set of parameters.
        This allows the model to jointly attend to information from different representation subspaces at different positions,
        enabling it to capture different types of dependencies and relationships within the input data.

            Parametres:
                `device`: torch.device, optional
                    device used for hardware acceleration.
                `d_model`: int, required (default=512)
                    the embed dimension of the model.
                `num_heads`: int, required (default=8)
                    specifies how many parallel attention heads will be used.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = self.d_model // self.num_heads

        self.attention = ScaledDotProductAttention()

        self.query_dense = nn.Linear(
            in_features=d_model, out_features=d_model, device=device
        )
        self.key_dense = nn.Linear(
            in_features=d_model, out_features=d_model, device=device
        )
        self.value_dense = nn.Linear(
            in_features=d_model, out_features=d_model, device=device
        )

        self.dense = nn.Linear(in_features=d_model, out_features=d_model, device=device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | Tensor | None,
    ) -> Tensor:
        """
        Forward pass for Multi-Head Attention Layer.

            Required Parameters:
                `query`: torch.Tensor; Tensor of shape [batch_size, sequence_length, d_model]
                `key`: torch.Tensor; Tensor of shape [batch_size, sequence_length, d_model]
                `value`: torch.Tensor; Tensor of shape [batch_size, sequence_length, d_model]
                `mask`: torch.Tensor; Tensor of shape [batch_size, 1, 1, sequence_length] (Padding Mask)
                or tensor of shape [batch_size, 1, 1, sequence_length, sequence_length] (Look-Ahead Mask)

            Returns:
                `output`: torch.Tensor; Tensor of shape [batch_size, sequence_length, d_model]
        """
        query, key, value = (
            self.query_dense(query),
            self.key_dense(key),
            self.value_dense(value),
        )
        query, key, value = self.split(x=query), self.split(x=key), self.split(x=value)

        output, _ = self.attention(query=query, key=key, value=value, mask=mask)
        output = self.concat(output)
        output = self.dense(output)

        return output

    def split(self, x: Tensor) -> Tensor:
        batch_size = x.size(dim=0)
        x = x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        return x

    def concat(self, x: Tensor) -> Tensor:
        batch_size = x.size(dim=0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return x
