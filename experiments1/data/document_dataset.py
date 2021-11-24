from pathlib import Path
from typing import Tuple
from typing import Union

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import Tensor

from data.tokenized_dataset import TokenizedDataset
from utils import preprocess_line, EMPTY_TOKEN


class DocumentDataset(TokenizedDataset):
    def __init__(self, path: Union[str, Path], tokenizer: Tokenizer, document_length: int = 300):
        super().__init__(path, tokenizer)

        self.document_length = document_length
        self.documents = []
        self.labels = []

        with open(self.path, 'r', encoding='UTF8') as f:
            lines = f.readlines()

        for line in lines:
            text, label = preprocess_line(line)
            self.documents.append(text)
            self.labels.append(label)

        self.classes = set(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.document_length > 0:
            text = torch.zeros(self.document_length) + self.tokenizer.token_to_id(EMPTY_TOKEN)
            document = self._tokenize_ids(self.documents[index])[:self.document_length]
            text[:len(document)] = torch.from_numpy(document)
        else:
            text = torch.from_numpy(self._tokenize_ids(self.documents[index]))

        return text, torch.from_numpy(np.array(self.labels[index]))

    @property
    def num_classes(self) -> int:
        return len(self.classes)
