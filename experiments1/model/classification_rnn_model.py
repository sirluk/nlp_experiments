from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model.classification_base_model import ClassificationBaseModel


class ClassificationRNNModel(ClassificationBaseModel):
    def __init__(
            self,
            embedding_weights: Tensor,
            num_classes: int,
            hidden_size: int = 512,
            num_layers: int = 2,
            dropout: float = .5,
            freeze_embeddings: bool = False,
            bidirectional: bool = False,
            reduction: str = 'last'
    ):
        super().__init__(embedding_weights, freeze_embeddings)

        self.reduction = reduction
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        if bidirectional:
            hidden_size *= 2

        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        inputs, input_lengths = inputs
        batch_size, max_length = inputs.shape

        # [batch_size, sequence_length, embedding_size]
        embedded = self.embedding(inputs.long().to(self.device))
        embedded = pack_padded_sequence(
            input=embedded,
            lengths=input_lengths,
            batch_first=True,
            enforce_sorted=False
        )

        output, _ = self.rnn(embedded)

        # [batch_size, sequence_length, hidden_size * num_directions]
        output, _ = pad_packed_sequence(output, batch_first=True)

        if self.reduction == 'mean':
            # since the sequence might be padded simply taking the mean does not work
            # [batch_size, hidden_size * num_directions]
            hidden = output.sum(1) / torch.repeat_interleave(Tensor(input_lengths), output.shape[-1]).reshape(batch_size, -1).to(self.device)
        elif self.reduction == 'last':
            if self.bidirectional:
                # [batch_size, sequence_length, num_directions, hidden_size]
                output = output.view(batch_size, max_length, 2, -1)

                # [batch_size, hidden_size * 2]
                hidden = torch.cat(
                    (
                        output[torch.arange(output.shape[0]), input_lengths.long() - 1, 0, :],  # last time step of forward direction
                        output[:, 0, 1, :]  # first time step of backward direction
                    ),
                    dim=-1
                )
            else:
                # [batch_size, hidden_size]
                hidden = output[torch.arange(output.shape[0]), input_lengths.long() - 1, :]
        else:
            raise ValueError(f'Reduction mode {self.reduction} is not supported.')

        return self.linear(hidden)
