import torch
import torch.nn as nn
from torch import Tensor, IntTensor, LongTensor


class CreatePaddingMask(nn.Module):

    def __init__(
        self,
        pad_token: int,
        device: torch.device = torch.device("mps"),
    ) -> None:
        """
        Parameters:
            `pad_token`: int, required
                value of the padding token.
        """
        super().__init__()
        self.pad_token = pad_token
        self.device = device

    def forward(self, x: IntTensor | LongTensor) -> Tensor:
        """
        Parameters:
            `x`: torch.IntTensor | torch.LongTensor; Tensor of shape [batch_size, sequence_length]

        Returns:
            `padding_mask`: torch.BoolTensor; Tensor of shape [batch_size, 1, 1, sequence_length]
        """
        padding_mask = (
            (x == self.pad_token)
            .unsqueeze(dim=1)
            .unsqueeze(dim=2)
            .type(torch.BoolTensor)
            .to(device=self.device)
        )

        return padding_mask
