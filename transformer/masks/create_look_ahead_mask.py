import torch
import torch.nn as nn
from torch import Tensor, IntTensor, LongTensor


class CreateLookAheadMask(nn.Module):

    def __init__(
        self,
        pad_token: int,
        device: torch.device = torch.device("mps"),
    ) -> None:
        """
        Create Look-ahead mask for decoder's first sublayer. Look-ahead mask also contains padding mask.
            Parameters:
                `pad_token`: int, required
                    value of the padding token.
                `device`: torch.device, optional
                    device used for hardware acceleration.

        """
        super().__init__()
        self.pad_token = pad_token
        self.device = device

    def forward(self, x: IntTensor | LongTensor) -> Tensor:
        """
        Parameters:
            `x`: torch.IntTensor | torch.LongTensor; Tensor of shape [batch_size, sequence_length]

        Returns:
            `look_ahead_mask`: torch.BoolTensor; Tensor of shape [batch_size, 1, sequence_length, sequence_length]

        """
        sequence_length = x.size(1)
        triangular_mask = (
            torch.triu(
                torch.ones(size=(sequence_length, sequence_length)).to(
                    device=self.device
                ),
                diagonal=1,
            )
            .type(dtype=torch.ByteTensor)
            .unsqueeze(dim=0)
            .unsqueeze(dim=1)
        )
        padding_mask = (
            (x == self.pad_token)
            .unsqueeze(dim=1)
            .unsqueeze(dim=2)
            .type(dtype=torch.ByteTensor)
        )
        look_ahead_mask = (
            torch.maximum(input=triangular_mask, other=padding_mask)
            .type(dtype=torch.BoolTensor)
            .to(device=self.device)
        )

        return look_ahead_mask
