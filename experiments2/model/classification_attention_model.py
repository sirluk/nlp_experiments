from typing import Tuple

from torch import Tensor, nn

from model import DotProductAttention, AdditiveAttention, MultiplicativeAttention
from model.backbone import Backbone
from model.classification_base_model import ClassificationBaseModel
from tensor_utils import generate_padding_mask


class ClassificationAttentionModel(ClassificationBaseModel):
    def __init__(
            self,
            embedding_weights: Tensor,
            num_classes: int,
            hidden_size: int = 512,
            num_layers: int = 2,
            dropout: float = .5,
            freeze_embeddings: bool = False,
            bidirectional: bool = True,
            attention_type: str = 'dot'
    ):
        super().__init__(embedding_weights, freeze_embeddings)

        self.backbone = Backbone(
            embedding_size=self.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        if attention_type == 'dot':
            self.attention = DotProductAttention(self.backbone.hidden_size)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(self.backbone.hidden_size)
        elif attention_type == 'multiplicative':
            self.attention = MultiplicativeAttention(self.backbone.hidden_size)
        else:
            raise ValueError(f'Attention of type {attention_type} is not supported')

        self.linear = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        inputs, input_lengths = inputs
        batch_size, sequence_length = inputs.shape

        # [batch_size, sequence_length, embedding_size]
        embedded = self.embedding(inputs.long().to(self.device))

        output = self.backbone(embedded, input_lengths)

        output, attention_weights = self.attention(output, generate_padding_mask(batch_size, sequence_length, input_lengths).to(self.device))

        return self.linear(output.squeeze(1)), attention_weights
