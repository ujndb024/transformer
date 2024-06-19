import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):

    def __init__(
        self,
        device: torch.device,
        d_model: int = 512,
        epsilon: float = 1e-12,
    ) -> None:
        """
        Layer Normalization, `Layernorm = (x + Sublayer(x))`.
            Parametres:
                `device`: torch.device, optional
                    device used for hardware acceleration.
                `d_model`: int, required (default=512)
                    the embed dimension of the model.
                `epsilon`: float, optional (default=1e-12)
                    small constant added to the variance to avoid division by zero.

            Objects Description:
                `self.gamma`: vector, where elements is filled only with 1.
                `self.beta`: vector, where elements is filled only with 0.
                `self.epsilon`: epsilon value.
        """
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(size=[d_model], device=device))
        self.beta = nn.Parameter(torch.zeros(size=[d_model], device=device))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Layer Normalization.

            Parameters:
                `x`: torch.Tensor; Tensor of shape [batch_size, sequence_length, d_model]
                    The input tensor to be normalized.

            Returns:
                `output`: torch.Tensor; Tensor of shape [batch_size, sequence_length, d_model]
                    The normalized tensor with the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        output = (x - mean) / torch.sqrt(var + self.epsilon)
        output = self.gamma * output + self.beta

        return output
