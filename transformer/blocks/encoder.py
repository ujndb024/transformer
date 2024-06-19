import torch
import torch.nn as nn
from torch import Tensor, IntTensor, LongTensor

from transformer.layers.encoder_layer import EncoderLayer
from transformer.embeddings.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        device: torch.device,
        max_len: int = 5000,
        d_model: int = 512,
        dff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters:
            `vocab_size`: int, required
                total size of the token vocabulary.
            `num_layers`: int, required
                the number of the encoder layers.
            `device`: torch.device, optional
                device used for hardware acceleration.
            `max_len`: int, required (default=5000)
                the maximum length of the incoming sequence.
            `d_model`: int, required (default=512)
                    the embed dimension of the model.
            `dff`: int, required (default=2048)
                dimension of the hidden layer.
            `num_heads`: int, required (default=8)
                    specifies how many parallel attention heads will be used.
            `dropout`: float, optional (default=0.1)
                the dropout value.
        """
        super().__init__()
        self.embedding = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=vocab_size,
            dropout=dropout,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    dff=dff,
                    num_heads=num_heads,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: IntTensor | LongTensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Forward pass for Encoder.

            Parameters:
                `x`: torch.IntTensor | torch.LongTensor; Tensor of shape [batch_size, encoder_sequence_length]
                    The input tensor containing token IDs.
                `mask`: torch.Tensor; Tensor of shape [batch_size, 1, 1, encoder_sequence_length]
                    The mask tensor to prevent attention to certain positions.

            Returns:
                `x`: torch.Tensor; Tensor of shape [batch_size, encoder_sequence_length, d_model]
                    The encoded representation of the input sequence.
        """
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x=x, mask=mask)

        return x
