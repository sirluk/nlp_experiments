import os
from pathlib import Path
from typing import Iterable, Union, Tuple, List, Callable

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFKC, BertNormalizer, Sequence
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, PackedSequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from metrics import accuracy
from model.classification_base_model import ClassificationBaseModel
from training_watcher import TrainingWatcher

UNK_TOKEN = '[UNK]'
EMPTY_TOKEN = '[EMPTY]'


def get_tokenizer(
        data: Union[Iterable[str], str, None] = None,
        data_path: Union[str, Path, None] = None,
        save_path: Union[str, Path, None] = None,
        vocab_size: int = 25_000
) -> Tokenizer:
    if save_path is not None and os.path.exists(save_path):
        tokenizer = Tokenizer.from_file(save_path)
        print(f'Loaded tokenizer. Vocabulary size: {tokenizer.get_vocab_size()}.')
        return tokenizer

    if data is None:
        data = []

        with open(data_path, 'r', encoding='UTF8') as f:
            for line in f.readlines():
                data.append(preprocess_line(line)[0])

    tokenizer = Tokenizer(WordPiece(vocab={ UNK_TOKEN: 1 }, unk_token=UNK_TOKEN))

    tokenizer.normalizer = Sequence(
        [
            NFKC(),
            BertNormalizer()
        ]
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = WordPieceDecoder()

    trainer = WordPieceTrainer(vocab_size=vocab_size, show_progress=True, special_tokens=[UNK_TOKEN, EMPTY_TOKEN])
    tokenizer.train_from_iterator(data, trainer=trainer)

    if save_path is not None:
        tokenizer.save(save_path)

    print(f'Trained tokenizer. Vocabulary size: {tokenizer.get_vocab_size()}.')

    return tokenizer


def preprocess_line(line: str) -> Tuple[str, int]:
    line = line.split(',', 1)[1]
    text, label = line.rsplit(',', 1)
    return text.strip('"'), int(label)


def get_initial_embedding(
        path: Union[Path, str],
        tokenizer: Tokenizer,
        save_path: Union[str, Path, None] = None
) -> nn.Embedding:
    if save_path is not None and os.path.exists(save_path):
        weight = np.load(save_path)
        embedding = nn.Embedding(weight.shape[0], weight.shape[1])
        embedding.weight = nn.Parameter(torch.from_numpy(weight))
        return embedding

    pre_trained = pd.read_csv(path, sep=" ", quoting=3, header=None, index_col=0)
    embedding_size = pre_trained.shape[1]
    embedding_weigths = torch.zeros(tokenizer.get_vocab_size(), embedding_size)

    not_contained = 0
    for word, index in tqdm(tokenizer.get_vocab().items()):
        if word in pre_trained.index:
            embedding_weigths[index] = torch.from_numpy(np.array(pre_trained.loc[word]))
        else:
            not_contained += 1
            embedding_weigths[index] = torch.rand(embedding_size)

    if save_path is not None:
        np.save(save_path, embedding_weigths.detach().numpy())

    print(f'Initialized {not_contained} token(s) randomly as they are not part of the pre-trained embeddings.')

    embedding = nn.Embedding(embedding_weigths.shape[0], embedding_weigths.shape[1])
    embedding.weight = nn.Parameter(embedding_weigths)

    return embedding


@torch.no_grad()
def nearest_neighbors(embedding: nn.Embedding, vocabulary: dict, input_word: str, k: int = 10) -> List[str]:
    distances = { }
    index = vocabulary[input_word]

    if index is None:
        return []

    for word, i in vocabulary.items():
        if i != index:
            distances[word] = torch.cdist(
                embedding(torch.from_numpy(np.array([index])).long()),
                embedding(torch.from_numpy(np.array([i])).long())
            ).item()

    return list(dict(sorted(distances.items(), key=lambda item: item[1])[:k]).keys())


def pad_collate(padding_value: int) -> Callable:
    def collate_fn(batch: List[Tensor]) -> Tuple[PackedSequence, Tensor, Tensor]:
        inputs, targets = zip(*batch)
        input_lengths = Tensor([len(x) for x in inputs])

        inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
        targets = torch.stack(targets)

        return inputs, targets, input_lengths

    return collate_fn


def train_and_evaluate(model: ClassificationBaseModel, name: str, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, weight_decay: float = 0.):
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        logger=Logger(
            model=model,
            log_dir=f'../runs/{name}'
        ),
        watcher=TrainingWatcher(max_epochs=40),
        validation_metrics={
            'accuracy': accuracy()
        }
    )

    _, metrics = model.evaluate(
        loader=test_loader,
        metrics={
            'accuracy': accuracy()
        }
    )

    acc = metrics['accuracy']
    #print(f'Test accuracy: {acc:.4f}')
    return acc
