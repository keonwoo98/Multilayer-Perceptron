"""Classification metrics for binary classification.

Predictions and targets are (n_samples, 2) arrays; the positive class
(malignant) is column 1.
"""

import numpy as np


def accuracy(y_pred, y_true):
    """Fraction of samples whose predicted class matches the true class."""
    return np.mean(y_pred.argmax(axis=1) == y_true.argmax(axis=1))


def precision_recall_f1(y_pred, y_true):
    """Precision, recall and F1 score for the positive (malignant) class."""
    pred = y_pred.argmax(axis=1)
    true = y_true.argmax(axis=1)
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1
