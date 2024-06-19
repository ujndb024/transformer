import torch
import torch.nn as nn
from torch import Tensor, IntTensor, LongTensor

from transformer.blocks.encoder import Encoder
from transformer.blocks.decoder import Decoder
from transformer.masks.create_padding_mask import CreatePaddingMask
from transformer.masks.create_look_ahead_mask import CreateLookAheadMask


class Transformer(nn.Module):

    def __init__(
        self,
        pad_token: int,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        num_layers: int,
        device: torch.device,
        d_model: int = 512,
        num_heads: int = 8,
        max_len: int = 5000,
        dff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """
        Parametres:
            `pad_token`: int, required
                value of the padding token.
            `encoder_vocab_size`: int, required
                size of the encoder vocabulary.
            `decoder_vocab_size`: int, required
                size of the decoder vocabulary.
            `num_layers`: int, required
                the number of the layers.
            `device`: torch.device, optional
                device used for hardware acceleration.
            `d_model`: int, required (default=512)
                the embed dimension of the model.
            `num_heads`: int, required (default=8)
                specifies how many parallel attention heads will be used.
            `max_len`: int, required (default=5000)
                the maximum length of the incoming sequence.
            `dff`: int, required (default=2048)
                dimension of the hidden layer.
            `dropout`: float, optional (default=0.1)
                the dropout value.
        """
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            dff=dff,
            vocab_size=encoder_vocab_size,
            dropout=dropout,
            num_layers=num_layers,
            device=device,
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            dff=dff,
            vocab_size=decoder_vocab_size,
            dropout=dropout,
            num_layers=num_layers,
            device=device,
        )
        self.padding_mask = CreatePaddingMask(pad_token=pad_token, device=device)
        self.look_ahead_mask = CreateLookAheadMask(pad_token=pad_token, device=device)

    def forward(
        self,
        encoder_input: IntTensor | LongTensor,
        decoder_input: IntTensor | LongTensor,
    ) -> Tensor:
        """
        Parameters:
            `encoder_input`: torch.IntTensor | torch.LongTensor; Tensor of shape [batch_size, encoder_sequence_length]
            `decoder_input` torch.IntTensor | torch.LongTensor; Tensor of shape [batch_size, decoder_sequence_length]

        Returns:
            `output`: torch.Tensor; Tensor of shape [batch_size, decoder_sequence_length, decoder_vocab_size]
        """
        padding_mask = self.padding_mask(encoder_input)
        look_ahead_mask = self.look_ahead_mask(decoder_input)

        encoder_output = self.encoder(x=encoder_input, mask=padding_mask)
        output = self.decoder(
            x=decoder_input,
            encoder_output=encoder_output,
            padding_mask=padding_mask,
            look_ahead_mask=look_ahead_mask,
        )

        return output
