from typing import Tuple

import torch
from torch import Tensor, nn

from model.backbone import Backbone
from model.classification_base_model import ClassificationBaseModel
from tensor_utils import generate_padding_mask


class ClassificationTransformerModel(ClassificationBaseModel):
    def __init__(
            self,
            embedding_weights: Tensor,
            num_classes: int,
            hidden_size: int = 512,
            num_layers: int = 2,
            dropout: float = .5,
            freeze_embeddings: bool = False,
            bidirectional: bool = True,
            num_heads: int = 1,
            num_transformer_layers: int = 1
    ):
        super().__init__(embedding_weights, freeze_embeddings)

        self.num_heads = num_heads

        self.backbone = Backbone(
            embedding_size=self.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(self.backbone.hidden_size, num_heads),
            num_layers=num_transformer_layers
        )
        self.linear = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        inputs, input_lengths = inputs
        batch_size, sequence_length = inputs.shape

        # [batch_size, sequence_length, embedding_size]
        embedded = self.embedding(inputs.long().to(self.device))

        output = self.backbone(embedded, input_lengths)

        output = self.transformer(output.permute(1, 0, 2), src_key_padding_mask=generate_padding_mask(batch_size, sequence_length, input_lengths).to(self.device))
        # use the output corresponding to the first time step
        return self.linear(output[0]), torch.zeros(0)  # return dummy value since we don't have access to the attention weights
