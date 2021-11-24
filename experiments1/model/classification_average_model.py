from typing import Tuple

import torch
from torch import Tensor, nn

from model.classification_base_model import ClassificationBaseModel


class ClassificationAverageModel(ClassificationBaseModel):
    def __init__(self, embedding_weights: Tensor, num_classes: int, freeze_embeddings: bool = False):
        super().__init__(embedding_weights, freeze_embeddings)

        self.linear = nn.Linear(self.embedding_size, num_classes)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        inputs, input_lengths = inputs
        batch_size, max_length = inputs.shape

        # [batch_size, sequence_length, embedding_size]
        embedded = self.embedding(inputs.long().to(self.device))

        # since the sequence might be padded simply taking the mean does not work
        feature_means = embedded.sum(1) / torch.repeat_interleave(Tensor(input_lengths), self.embedding_size).reshape(batch_size, -1).to(self.device)
        return self.linear(feature_means)
