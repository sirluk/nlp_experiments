from typing import Tuple, Union, List

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from training_watcher import TrainingWatcher


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: Optimizer,
            watcher: TrainingWatcher,
            logger: Logger,
            validation_metrics: Union[dict, None] = None
    ):
        if validation_metrics is None:
            validation_metrics = { }

        self.evaluate(
            loader=val_loader,
            logger=logger,
            metrics=validation_metrics
        )

        while True:
            self.__step(
                loader=train_loader,
                optimizer=optimizer,
                logger=logger
            )
            validation_loss, _ = self.evaluate(
                loader=val_loader,
                logger=logger,
                metrics=validation_metrics
            )

            if watcher.should_stop_training(validation_loss):
                break

    def __step(self, loader: DataLoader, optimizer: Optimizer, logger: Logger) -> float:
        self.train()
        average_loss = 0.

        progress_bar = tqdm(
            loader,
            ascii=' -=',
            bar_format='{n_fmt}/{total_fmt} [{bar:40}{bar:-40b}] - {elapsed}s - average loss: {postfix[0][average_loss]:.3f} - batch loss: {postfix[0][batch_loss]:.3f}',
            postfix=[dict(average_loss=0, batch_loss=0)]
        )

        for i, batch in enumerate(progress_bar):
            inputs, targets = self.__unpack_batch(batch)

            optimizer.zero_grad()
            loss, _ = self._calculate_loss(inputs, targets)
            loss.backward()
            optimizer.step()

            loss = loss.item()
            logger.batch_loss(loss)
            average_loss += (loss - average_loss) / (i + 1)

            progress_bar.postfix[0]['average_loss'] = average_loss
            progress_bar.postfix[0]['batch_loss'] = loss
            progress_bar.update()

        logger.epoch_loss(average_loss)

        return average_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, metrics: dict, logger: Union[Logger, None] = None) -> Tuple[float, dict]:
        self.eval()

        average_loss = 0.
        predictions = None
        targets = None

        for i, batch in enumerate(loader):
            x, y = self.__unpack_batch(batch)

            loss, y_hat = self._calculate_loss(x, y)
            average_loss += (loss.item() - average_loss) / (i + 1)

            if predictions is None:
                predictions = y_hat.cpu().numpy()
                targets = y.numpy()
            else:
                predictions = np.concatenate((predictions, y_hat.cpu().numpy()))
                targets = np.concatenate((targets, y.numpy()))

        metric_values = { name: metric(predictions, targets) for name, metric in metrics.items() }

        if logger is not None:
            logger.validation_loss(
                loss=average_loss,
                metrics=metric_values
            )

        return average_loss, metric_values

    @staticmethod
    def __unpack_batch(batch: List) -> Tuple[Union[Tuple, Tensor], Tensor]:
        if len(batch) == 3:  # padded sequence
            inputs, targets, input_lengths = batch
            inputs = (inputs, input_lengths)
        else:
            inputs, targets = batch

        return inputs, targets

    def _calculate_loss(self, inputs: Union[Tuple, Tensor], targets: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()
