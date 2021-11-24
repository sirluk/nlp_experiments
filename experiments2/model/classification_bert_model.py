from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertModel

from model.base_model import BaseModel
from tensor_utils import generate_bert_padding_mask


class ClassificationBERTModel(BaseModel):
    def __init__(
            self,
            num_classes: int,
            name: str = 'bert-base-cased'
    ):
        super().__init__()

        self.model = BertModel.from_pretrained(name, return_dict=False, output_attentions=True)
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        inputs, input_lengths = inputs
        batch_size, sequence_length = inputs.shape
        _, output, attention_weights = self.model(input_ids=inputs.int().to(self.device), attention_mask=generate_bert_padding_mask(batch_size, sequence_length, input_lengths).to(self.device))
        return self.linear(output), attention_weights

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def _calculate_loss(self, inputs: Union[Tuple, Tensor], targets: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        logits, _ = self(inputs)
        return F.cross_entropy(logits, targets.long().to(self.device)), logits

    def predict(self, x: Tensor) -> Tensor:
        logits, _ = self(x)
        return torch.softmax(logits, dim=-1)
