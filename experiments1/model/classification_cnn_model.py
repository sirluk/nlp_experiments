from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.classification_base_model import ClassificationBaseModel


class ClassificationCNNModel(ClassificationBaseModel):
    def __init__(
            self,
            embedding_weights: Tensor,
            num_classes: int,
            num_kernels: Tuple = (512,512,512),
            window_sizes: Tuple = (1, 2, 3),
            freeze_embeddings: bool = False,
            dropout: Union[None, float] = None
    ):
        super().__init__(embedding_weights, freeze_embeddings)

        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=kernels,
                    kernel_size=(window_size, self.embedding_size),
                    padding=(window_size - 1, 0)
                ) for kernels, window_size in zip(num_kernels, window_sizes)
            ]
        )

        self.linear = nn.Linear(sum(num_kernels), num_classes)

        if dropout is None:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        inputs, input_lengths = inputs
        batch_size, max_length = inputs.shape

        # [batch_size, 1, sequence_length, embedding_size]
        embedded = self.embedding(inputs.long().to(self.device)).unsqueeze(1)

        # [batch_size, sum(kernel_size)]
        convolved = torch.cat(
            [
                # [batch_size, kernel_size]
                F.max_pool1d(
                    # [batch_size, kernel_size, sequence_length]
                    F.relu(conv(embedded)).squeeze(-1),
                    kernel_size=max_length
                ).squeeze(-1) for conv in self.convolutions
            ],
            dim=-1
        )

        return self.linear(self.dropout(convolved))
