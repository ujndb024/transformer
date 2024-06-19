import torch
import torch.nn as nn
from torch import Tensor, IntTensor, LongTensor

import math


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        device: torch.device,
        d_model: int = 512,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        """
        `PositionalEncoding`, which is used in order to get the information of the word's position.
        It is calculated by adding all embedding vectors with positional information.
            Parametres:
                `device`: torch.device, optional
                    device used for hardware acceleration.
                `d_model`: int, required
                    the embed dimension of the model.
                `max_len`: int, required (default=5000)
                    the maximum length of the incoming sequence.
                `dropout`: float, optional (default=0.1)
                    the dropout value.
            Examples:
                >>> pos_encoder = PositionalEncoding(d_model)
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.pos_encoding = torch.zeros(size=[max_len, d_model], device=device)
        self.pos_encoding.requires_grad = False

        positions_list = torch.arange(
            start=0,
            end=max_len,
            dtype=torch.float,
            device=device,
        ).unsqueeze(dim=1)
        division_term = torch.exp(
            torch.arange(start=0, end=d_model, step=2, device=device).float()
            * (-math.log(10000.0) / d_model)
        )

        self.pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        self.pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

    def forward(self, x: IntTensor | LongTensor) -> Tensor:
        """
        Forward pass of Positional Encoding.
            Parameters:
                `x`: torch.IntTensor | torch.LongTensor; Tensor of shape [batch_size, sequence_length]

            Returns:
                `output`: torch.Tensor; Tensor of shape [1, sequence_length, d_model]
        """
        _, sequence_length = x.size()
        x = self.pos_encoding[:sequence_length, :]
        x = x.unsqueeze(dim=0)
        output = self.dropout(x)

        return output
