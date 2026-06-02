"""Loss functions measuring how wrong a prediction is."""

import numpy as np

# Tiny constant added inside log() to avoid log(0) = -infinity.
_EPS = 1e-15


def categorical_cross_entropy(y_pred, y_true):
    """Average cross-entropy between predictions and one-hot targets.

    y_pred, y_true: arrays of shape (n_samples, n_classes).
    For each sample only the true class term survives (y_true is one-hot),
    so this rewards putting high probability on the correct class.
    """
    y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def binary_cross_entropy(y_pred, y_true):
    """Binary cross-entropy used to evaluate the final prediction.

    y_pred, y_true: 1-D arrays of probabilities/labels for the positive class.
        E = -(1/N) * sum[ y*log(p) + (1-y)*log(1-p) ]
    """
    y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
