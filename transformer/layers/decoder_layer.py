import torch
import torch.nn as nn
from torch import Tensor

from transformer.sublayers.layer_norm import LayerNorm
from transformer.sublayers.multi_head_attention import MultiHeadAttention
from transformer.sublayers.position_wise_feed_forward import (
    PositionWiseFeedForward,
)


class DecoderLayer(nn.Module):

    def __init__(
        self,
        device: torch.device,
        d_model: int = 512,
        dff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters:
            `device`: torch.device, optional
                device used for hardware acceleration.
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
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, device=device
        )
        self.layer_norm1 = LayerNorm(d_model=d_model, device=device)
        self.dropout1 = nn.Dropout(p=dropout)

        self.encoder_decoder_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, device=device
        )
        self.layer_norm2 = LayerNorm(d_model=d_model, device=device)
        self.dropout2 = nn.Dropout(p=dropout)

        self.ffnn = PositionWiseFeedForward(
            d_model=d_model, dff=dff, dropout=dropout, device=device
        )
        self.layer_norm3 = LayerNorm(d_model=d_model, device=device)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        look_ahead_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass for Decoder Layer.

            Required Parameters:
                `x`: torch.Tensor; Tensor of shape [batch_size, decoder_sequence_length, d_model]
                    The input tensor to the decoder layer.
                `encoder_output`: torch.Tensor; Tensor of shape [batch_size, encoder_sequence_length, d_model]
                    The output tensor from the encoder.
                `look_ahead_mask`: torch.Tensor; Tensor of shape [batch_size, 1, decoder_sequence_length, decoder_sequence_length]
                    The look-ahead mask to prevent attention to future tokens for decoder's input.
                `padding_mask`: torch.Tensor; Tensor of shape [batch_size, 1, 1, encoder_sequence_length]
                    The padding mask to mask out padding tokens in the encoder's output.

            Returns:
                `x`: torch.Tensor; Tensor of shape [batch_size, decoder_sequence_length, d_model].
        """
        original_x = x
        x = self.layer_norm1(x)
        x = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=look_ahead_mask,
        )
        sublayer_x = self.dropout1(x)

        x = original_x + sublayer_x

        if encoder_output is not None:
            original_x = x
            x = self.layer_norm2(x)
            encoder_output = self.layer_norm2(encoder_output)
            x = self.encoder_decoder_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                mask=padding_mask,
            )
            sublayer_x = self.dropout2(x)

            x = original_x + sublayer_x

        original_x = x
        x = self.layer_norm3(x)
        x = self.ffnn(x)
        sublayer_x = self.dropout3(x)

        x = original_x + sublayer_x

        return x
