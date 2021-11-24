from typing import Tuple, Union

import torch
from torch import Tensor, nn

from model.base_model import BaseModel
import torch.nn.functional as F


class ClassificationBaseModel(BaseModel):
    def __init__(self, embedding_weights: Tensor, freeze_embeddings: bool):
        super().__init__()

        self.vocab_size, self.embedding_size = embedding_weights.shape

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=not freeze_embeddings)

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _calculate_loss(self, inputs: Tuple[Tensor, Tensor], targets: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        logits, _ = self(inputs)
        return F.cross_entropy(logits, targets.long().to(self.device)), logits

    def predict(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        logits, _ = self(x)
        return torch.softmax(logits, dim=-1)
