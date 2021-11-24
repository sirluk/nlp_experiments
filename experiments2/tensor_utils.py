import torch
from torch import Tensor


def generate_padding_mask(batch_size: int, sequence_length: int, input_lengths: Tensor) -> Tensor:
    mask = torch.zeros(batch_size, sequence_length).bool()
    for i, input_length in enumerate(input_lengths):
        mask[i, int(input_length):] = True
    return mask


def generate_bert_padding_mask(batch_size: int, sequence_length: int, input_lengths: Tensor) -> Tensor:
    mask = torch.ones(batch_size, sequence_length).long()
    for i, input_length in enumerate(input_lengths):
        mask[i, int(input_length):] = 0
    return mask
