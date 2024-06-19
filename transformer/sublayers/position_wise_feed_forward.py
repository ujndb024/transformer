import torch
import torch.nn as nn
from torch import Tensor


class PositionWiseFeedForward(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        dff: int = 2048,
        dropout: float = 0.1,
        device: torch.device = torch.device("mps"),
    ) -> None:
        """
        PositionWiseFeedForward allows the model to capture complex patterns and relationships within each position of the input sequence,
        helping in tasks like sequence modeling, language translation, and text generation.

            Parametres:
                `d_model`: int, required (default=512)
                    the embed dimension of the model.
                `dff`: int, required (default=2048)
                    dimension of the hidden layer.
                `dropout`: float, optional (default=0.1)
                    the dropout value.
        """
        super().__init__()
        self.weight_tensor1 = nn.Linear(
            in_features=d_model, out_features=dff, device=device
        )
        self.weight_tensor2 = nn.Linear(
            in_features=dff, out_features=d_model, device=device
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Position-wise Feed Forward layer.

            Parameters:
                `x`: torch.Tensor; Tensor of shape [sequence_length, batch_size, d_model].
                    The input tensor of the position-wise feed forward layer.

            Returns:
                `x`: torch.Tensor; Tensor of shape [sequence_length, batch_size, d_model].
                    Represents the transformed input sequence after passing through the feed-forward layer.
        """
        x = self.weight_tensor1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.weight_tensor2(x)

        return x
