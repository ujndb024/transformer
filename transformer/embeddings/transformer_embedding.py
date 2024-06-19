import torch
import torch.nn as nn
from torch import Tensor, IntTensor, LongTensor

from transformer.embeddings.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        device: torch.device,
        max_len: int = 5000,
        d_model: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """
        Combines `PositionalEncoding` with `nn.Embedding`.
            Parameters:
                `vocab_size`: int, required
                    total size of the token vocabulary.
                `device`: torch.device, optional
                    device used for hardware acceleration.
                `max_len`: int, required (default=5000)
                    the maximum length of the incoming sequence.
                `d_model`: int, required (default=512)
                    the embed dimension of the model.
                `dropout`: float, optional (default=0.5)
                    the dropout value.

        """
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_len, device=device
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: IntTensor | LongTensor) -> Tensor:
        """
        Parametres:
            `x`: torch.IntTensor | torch.LongTensor; Tensor of shape [batch_size, sequence_length]
                The sequence which will be fed to the `nn.Embedding` and `PositionalEncoding`.

        Returns:
            `output`: torch.Tensor; Tensor of shape [batch_size, sequence_length, d_model]
        """
        token_embedding = self.token_embedding(x)
        positional_encoding = self.positional_encoding(x)
        output = self.dropout(input=token_embedding + positional_encoding)

        return output
