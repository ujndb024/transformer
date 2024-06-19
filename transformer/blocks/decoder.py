import torch
import torch.nn as nn
from torch import Tensor, IntTensor, LongTensor

from transformer.layers.decoder_layer import DecoderLayer
from transformer.embeddings.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):

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
        Parametres:
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
                DecoderLayer(
                    d_model=d_model,
                    dff=dff,
                    num_heads=num_heads,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(
            in_features=d_model, out_features=vocab_size, device=device
        )

    def forward(
        self,
        x: IntTensor | LongTensor,
        encoder_output: Tensor,
        look_ahead_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass for Decoder.

            Required Parameters:
                `x`: torch.IntTensor | torch.LongTensor; Tensor [batch_size, decoder_sequence_length]
                    The input tensor containing token IDs.
                `encoder_output`: torch.Tensor; Tensor [batch_size, encoder_sequence_length, d_model]
                    The output tensor from the encoder.
                `look_ahead_mask`: torch.Tensor; Tensor [batch_size, 1, decoder_sequence_length, decoder_sequence_length]
                    The look-ahead mask to prevent attention to future tokens.
                `padding_mask`: torch.Tensor; Tensor [batch_size, 1, 1, decoder_sequence_length]
                    The padding mask to mask out padding tokens in the encoder output.

            Returns:
                `output`: torch.Tensor; Tensor of shape [batch_size, decoder_sequence_length, d_model].
        """

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )

        output = self.linear(x)

        return output
