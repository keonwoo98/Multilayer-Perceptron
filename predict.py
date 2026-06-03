"""Load a trained model and evaluate it on a dataset.

Usage:
    python predict.py --data data_valid.csv
"""

import argparse
import sys

import numpy as np

from srcs.network import Network
from srcs.losses import binary_cross_entropy
from srcs.metrics import accuracy, precision_recall_f1
from srcs.preprocessing import load_data, standardize, one_hot

MODEL_PATH = "models/model.npz"


def load_model(path):
    """Rebuild the network from a saved .npz file and return it with stats."""
    data = np.load(path)
    layer_sizes = data["layer_sizes"]
    net = Network(layer_sizes)
    for i, layer in enumerate(net.layers):
        layer.W = data[f"W{i}"]
        layer.b = data[f"b{i}"]
    return net, data["mean"], data["std"]


def main():
    parser = argparse.ArgumentParser(description="Predict and evaluate with a saved model.")
    parser.add_argument("--data", default="data_valid.csv", help="csv to evaluate on")
    args = parser.parse_args()

    try:
        net, mean, std = load_model(MODEL_PATH)
    except FileNotFoundError:
        sys.exit(f"error: model not found: {MODEL_PATH}. Train a model first with train.py")

    # Apply the SAME normalization that training used.
    try:
        X, y_raw = load_data(args.data)
    except FileNotFoundError:
        sys.exit(f"error: dataset not found: {args.data}")
    X = standardize(X, mean, std)
    y = one_hot(y_raw)

    preds = np.array([net.forward(xi) for xi in X])

    # Binary cross-entropy on the malignant-class probability (column 1).
    bce = binary_cross_entropy(preds[:, 1], y[:, 1])
    acc = accuracy(preds, y)
    precision, recall, f1 = precision_recall_f1(preds, y)

    print(f"samples evaluated : {len(X)}")
    print(f"binary cross-entropy loss : {bce:.4f}")
    print(f"accuracy          : {acc * 100:.2f}%")
    print(f"precision         : {precision:.4f}")
    print(f"recall            : {recall:.4f}")
    print(f"f1 score          : {f1:.4f}")


if __name__ == "__main__":
    main()
