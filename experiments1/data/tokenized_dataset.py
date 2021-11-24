from abc import ABC
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset


class TokenizedDataset(Dataset, ABC):
    def __init__(self, path: Union[str, Path], tokenizer: Tokenizer):
        self.path = path
        self.tokenizer = tokenizer

    def _one_hot_tokenize(self, text: str) -> Tensor:
        return self._one_hot_encode(self._tokenize_ids(text))

    def _one_hot_encode(self, ids: Union[np.ndarray, int]) -> Tensor:
        if not isinstance(ids, np.ndarray):
            ids = np.array(ids)

        return F.one_hot(torch.from_numpy(ids).long(), num_classes=self.tokenizer.get_vocab_size())

    def _tokenize_ids(self, text: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(text).ids)

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens
