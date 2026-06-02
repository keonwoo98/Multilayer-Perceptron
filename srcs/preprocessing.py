"""Load the dataset and prepare it for training.

Raw data.csv has no header. The columns are:
  - column 0: id (dropped, irrelevant to diagnosis)
  - column 1: diagnosis ('M' malignant / 'B' benign), the label
  - columns 2-31: 30 numeric features
"""

import numpy as np
import pandas as pd

# The 10 base measurements, each given as mean / standard-error / worst.
_BASE = ["radius", "texture", "perimeter", "area", "smoothness",
         "compactness", "concavity", "concave_points", "symmetry", "fractal_dim"]
COLUMN_NAMES = ["id", "diagnosis"] + [
    f"{base}_{stat}" for stat in ("mean", "se", "worst") for base in _BASE
]


def load_data(path):
    """Read the csv and split it into features (X) and labels (y).

    Returns:
        X: float array of shape (n_samples, 30)
        y: array of 'M'/'B' strings of shape (n_samples,)
    """
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    y = df["diagnosis"].values
    X = df.drop(columns=["id", "diagnosis"]).values
    return X, y


def compute_stats(X):
    """Compute the per-feature mean and standard deviation from X.

    Called on the training set only; the returned stats are reused to
    standardize validation/test data the same way.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std


def standardize(X, mean, std):
    """Rescale each feature to mean 0 and standard deviation 1."""
    return (X - mean) / std


def one_hot(y):
    """Encode labels as two-column targets: B -> [1, 0], M -> [0, 1]."""
    encoded = np.zeros((len(y), 2))
    for i, label in enumerate(y):
        if label == "M":
            encoded[i] = [0, 1]
        else:
            encoded[i] = [1, 0]
    return encoded
