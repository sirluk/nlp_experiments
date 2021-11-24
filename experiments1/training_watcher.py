from typing import Union


class TrainingWatcher:
    def __init__(self, patience: int = 2, delta: float = 0., max_epochs: Union[int, None] = None):
        self.patience = patience
        self.delta = delta
        self.max_epochs = max_epochs
        self.epochs = 0

        self.best_validation_loss = None
        self.counter = 0

    def should_stop_training(self, validation_loss: float) -> bool:
        self.epochs += 1

        if self.max_epochs is not None and self.epochs >= self.max_epochs:
            return True

        if self.best_validation_loss is None:
            self.best_validation_loss = validation_loss
        elif validation_loss > self.best_validation_loss + self.delta:
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best_validation_loss = validation_loss
            self.counter = 0

        return False
