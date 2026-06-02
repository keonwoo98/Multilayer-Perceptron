"""Train the multilayer perceptron and save the model to disk.

Usage:
    python train.py --train data_train.csv --valid data_valid.csv \
        --layers 24 24 --epochs 70 --lr 0.1
"""

import argparse
import os

import numpy as np

from srcs.network import Network
from srcs.preprocessing import load_data, compute_stats, standardize, one_hot

MODEL_PATH = "models/model.npz"


def main():
    parser = argparse.ArgumentParser(description="Train the MLP and save the model.")
    parser.add_argument("--train", default="data_train.csv", help="training csv")
    parser.add_argument("--valid", default="data_valid.csv", help="validation csv")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 24],
                        help="sizes of the hidden layers (default: 24 24)")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load and preprocess. Stats come from the training set only.
    X_train, y_train_raw = load_data(args.train)
    X_val, y_val_raw = load_data(args.valid)
    mean, std = compute_stats(X_train)
    X_train = standardize(X_train, mean, std)
    X_val = standardize(X_val, mean, std)
    y_train = one_hot(y_train_raw)
    y_val = one_hot(y_val_raw)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_val.shape}")

    # Input size is fixed by the data (30); output is 2 (benign/malignant).
    layer_sizes = [X_train.shape[1]] + args.layers + [2]
    net = Network(layer_sizes, seed=args.seed)
    net.fit(X_train, y_train, X_val, y_val, args.epochs, args.lr)

    # Save weights, network shape, and the normalization stats together,
    # so predict.py can rebuild the exact same model and transform.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    params = {"layer_sizes": np.array(layer_sizes), "mean": mean, "std": std}
    for i, layer in enumerate(net.layers):
        params[f"W{i}"] = layer.W
        params[f"b{i}"] = layer.b
    np.savez(MODEL_PATH, **params)
    print(f"> saving model '{MODEL_PATH}' to disk...")


if __name__ == "__main__":
    main()
