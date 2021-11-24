import numpy as np


def accuracy():
    def metric(y_hat: np.ndarray, y: np.ndarray) -> float:
        return (y_hat.argmax(-1) == y).sum() / y_hat.shape[0]

    return metric
