from collections import defaultdict
from pathlib import Path
from time import time
from typing import Union, List

import matplotlib.pyplot as plt
import torch
from IPython.core.display import clear_output
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: Union[str, Path], model: nn.Module):
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self.model = model

        self.writer = SummaryWriter(log_dir / 'tensorboard')

        self.dump_dir = log_dir / 'dump'
        self.dump_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoint_dir = log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.values = defaultdict(lambda: defaultdict(list))
        self.dump_idx = defaultdict(lambda: defaultdict(lambda: -1))

        self.steps = 0
        self.epochs = 0
        self.best_validation_loss = None

    @property
    def best_checkpoint_path(self) -> str:
        return str(self.checkpoint_dir / f'best.pt')

    def validation_loss(self, loss: float, metrics: dict):
        clear_output(wait=True)

        self.__add_scalar('val', 'loss', loss, self.epochs)

        for name, value in metrics.items():
            self.__add_scalar('val', name, value, self.epochs)

        torch.save(self.model.state_dict(), self.checkpoint_dir / f'{self.epochs}.pt')

        if self.best_validation_loss is None or self.best_validation_loss > loss:
            self.best_validation_loss = loss
            torch.save(self.model.state_dict(), self.best_checkpoint_path)

        self.__dump()

        metrics_string = ', '.join(f'{name}: {val:.3f}' for name, val in metrics.items())
        if metrics_string != '':
            metrics_string = f' ({metrics_string})'

        print(f'Epoch {self.epochs}. Validation loss: {loss:.3f}{metrics_string}')

        if self.epochs > 0:
            self.plot_metrics()

        self.epochs += 1

    def epoch_loss(self, loss: float):
        self.__add_scalar('train', 'epoch_loss', loss, self.epochs)

    def batch_loss(self, loss: float):
        self.steps += 1
        self.__add_scalar('train', 'batch_loss', loss, self.steps)

    def __add_scalar(self, tag: str, name: str, value: float, step: int):
        self.writer.add_scalar(f'{tag}/{name}', value, step)
        self.values[tag][name].append((step, time(), value))

    def __dump(self):
        for tag, names in self.values.items():
            for name, rows in names.items():
                with open(self.dump_dir / f'{tag}_{name}.log', 'a') as f:
                    for i, row in enumerate(rows):
                        if i > self.dump_idx[tag][name]:
                            f.write(f"{','.join([str(x) for x in row])}\n")
                            self.dump_idx[tag][name] = i

    def plot_metrics(self):
        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(121, label='train')
        ax1.plot(self.get_values('train', 'batch_loss'), color='C0')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Update (Training)', color='C0')
        ax1.xaxis.grid(False)
        ax1.set_ylim(0)

        ax2 = fig.add_subplot(121, label='val', frame_on=False)
        ax2.plot(self.get_values('val', 'loss'), color='C1')
        ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        ax2.set_xlabel('Epoch (Validation)', color='C1')
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.grid(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set_ylim(ax1.get_ylim())

        accuracy_values = self.get_values('val', 'accuracy')
        if accuracy_values is not None:
            ax3 = fig.add_subplot(122, label='acc')
            ax3.plot(accuracy_values, color='C2')
            ax3.set_xlabel('Epoch (Validation)', color='C2')
            ax3.set_ylabel('Accuracy')
            ax3.tick_params(axis='x')
            ax3.tick_params(axis='y')
            ax3.xaxis.grid(False)
            ax3.set_ylim((0, 1))

        fig.tight_layout(pad=2)
        plt.show()

    def get_values(self, tag: str, name: str) -> Union[List, None]:
        if tag in self.values:
            if name in self.values[tag]:
                return [x[2] for x in self.values[tag][name]]

        return None
