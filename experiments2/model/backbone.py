from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Backbone(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool
    ):
        super().__init__()

        if hidden_size > 0:
            self.backbone = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

            if bidirectional:
                self.hidden_size = hidden_size * 2
        else:
            self.hidden_size = self.embedding_size

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        if self.backbone is not None:
            embedded = pack_padded_sequence(
                input=inputs,
                lengths=input_lengths,
                batch_first=True,
                enforce_sorted=False
            )

            output, _ = self.backbone(embedded)

            # [batch_size, sequence_length, hidden_size * num_directions]
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output = inputs

        return output
