from pathlib import Path
from typing import Tuple, List
from typing import Union

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer

from model import ClassificationBaseModel

UNK_TOKEN = '[UNK]'
PAD_TOKEN = '[PAD]'


class DocumentDataset(Dataset):
    def __init__(self, path: Union[str, Path], tokenizer: Tokenizer, document_length: int = 300):
        super().__init__()

        self.document_length = document_length
        self.documents = []
        self.labels = []
        self.path = path
        self.tokenizer = tokenizer

        with open(self.path, 'r', encoding='UTF8') as f:
            lines = f.readlines()

        for line in lines:
            text, label = self.__preprocess_line(line)
            self.documents.append(text)
            self.labels.append(label)

        self.classes = set(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.document_length > 0:
            token_ids = torch.zeros(self.document_length) + self._tokenize_ids(PAD_TOKEN)[0]
            document = self._tokenize_ids(self.documents[index])

            if document.shape[0] > self.document_length:
                document = document[:self.document_length]

            token_ids[:document.shape[0]] = torch.from_numpy(document)
        else:
            token_ids = torch.from_numpy(self._tokenize_ids(self.documents[index]))

        return token_ids, torch.from_numpy(np.array(self.labels[index]))

    @staticmethod
    def __preprocess_line(line: str) -> Tuple[str, int]:
        line = line.split(',', 1)[1]
        text, label = line.rsplit(',', 1)
        return text.strip('"'), int(label)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _tokenize_ids(self, text: str) -> np.ndarray:
        if isinstance(self.tokenizer, BertTokenizer):
            return np.array(self.tokenizer.encode(text, max_length=self.document_length, truncation=True, padding=True))
        else:
            return np.array(self.tokenizer.encode(text).ids)

    def find_examples(self, model: ClassificationBaseModel, k: int = 2, seed: int = 42) -> Tuple[List[int], List[int]]:
        correct = []
        misclassified = []

        indices = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(indices)

        for i in indices:
            token_ids, label = self[i]
            prediction = model.predict((token_ids.unsqueeze(0), Tensor([token_ids.shape[-1]]))).argmax()

            if len(correct) < k and prediction.item() == label:
                correct.append(i)
            elif len(misclassified) < k:
                misclassified.append(i)

            if len(misclassified) == k and len(correct) == k:
                break

        return correct, misclassified
