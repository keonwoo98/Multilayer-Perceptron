"""Load a trained model and evaluate it on a dataset.

Usage:
    python predict.py --data data_valid.csv
"""

import argparse

import numpy as np

from srcs.network import Network
from srcs.losses import binary_cross_entropy
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

    net, mean, std = load_model(MODEL_PATH)

    # Apply the SAME normalization that training used.
    X, y_raw = load_data(args.data)
    X = standardize(X, mean, std)
    y = one_hot(y_raw)

    preds = np.array([net.forward(xi) for xi in X])

    # Accuracy: predicted class vs true class.
    accuracy = np.mean(preds.argmax(axis=1) == y.argmax(axis=1))

    # Binary cross-entropy on the malignant-class probability (column 1).
    bce = binary_cross_entropy(preds[:, 1], y[:, 1])

    print(f"samples evaluated : {len(X)}")
    print(f"accuracy          : {accuracy * 100:.2f}%")
    print(f"binary cross-entropy loss : {bce:.4f}")


if __name__ == "__main__":
    main()
