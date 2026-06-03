"""Train the multilayer perceptron and save the model to disk.

Usage:
    python train.py data_training.csv
    python train.py data_training.csv --layers 24 24 --epochs 80 --lr 0.3
    python train.py data_train.csv --valid data_valid.csv --optimizer adam --lr 0.001

If no --valid file is given, the training set is split internally so the
program also works with the single file produced by evaluation.py.
"""

import argparse
import os
import sys

import numpy as np

from srcs.network import Network
from srcs.optimizer import SGD, Adam
from srcs.preprocessing import (
    load_data, compute_stats, standardize, one_hot, train_val_split,
)
from srcs.plotting import plot_history

MODEL_PATH = "models/model.npz"


def main():
    parser = argparse.ArgumentParser(description="Train the MLP and save the model.")
    parser.add_argument("dataset", help="training csv (e.g. data_training.csv)")
    parser.add_argument("--valid", default=None,
                        help="optional separate validation csv; if omitted, the "
                             "training set is split internally")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 24],
                        help="hidden layer sizes (default: 24 24)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.3,
                        help="learning rate (use ~0.001 for adam)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="L2 regularization strength (curbs overfitting)")
    parser.add_argument("--no_early_stopping", dest="early_stopping",
                        action="store_false",
                        help="disable early stopping (enabled by default; it stops "
                             "training when validation loss plateaus and restores the "
                             "best weights, which improves single-run robustness)")
    parser.add_argument("--patience", type=int, default=15,
                        help="epochs to wait before early stopping")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="validation fraction when no --valid is given")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        X, y_raw = load_data(args.dataset)
    except FileNotFoundError:
        sys.exit(f"error: dataset not found: {args.dataset}")

    if args.valid:
        try:
            X_val, y_val_raw = load_data(args.valid)
        except FileNotFoundError:
            sys.exit(f"error: validation set not found: {args.valid}")
        X_train, y_train_raw = X, y_raw
    else:
        X_train, y_train_raw, X_val, y_val_raw = train_val_split(
            X, y_raw, args.val_ratio, args.seed)

    # Normalization stats come from the training part only.
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

    if args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(lr=args.lr, weight_decay=args.weight_decay)

    history = net.fit(X_train, y_train, X_val, y_val, args.epochs, optimizer,
                      batch_size=args.batch_size,
                      early_stopping=args.early_stopping, patience=args.patience)

    # Save weights, network shape, and the normalization stats together,
    # so predict.py can rebuild the exact same model and transform.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    params = {"layer_sizes": np.array(layer_sizes), "mean": mean, "std": std}
    for i, layer in enumerate(net.layers):
        params[f"W{i}"] = layer.W
        params[f"b{i}"] = layer.b
    np.savez(MODEL_PATH, **params)
    print(f"> saving model '{MODEL_PATH}' to disk...")

    plot_history(history)


if __name__ == "__main__":
    main()
