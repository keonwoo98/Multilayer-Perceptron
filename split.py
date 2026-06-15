"""Split data.csv into a training set and a validation set.

Usage:
    python split.py data.csv --val_ratio 0.2 --seed 42
"""

import argparse

import pandas as pd

from srcs.preprocessing import split_indices


def split_data(path, val_ratio, seed):
    """Shuffle the rows and cut them into a training and a validation set."""
    df = pd.read_csv(path, header=None)
    train_idx, valid_idx = split_indices(len(df), val_ratio, seed)
    return df.iloc[train_idx], df.iloc[valid_idx]


def main():
    parser = argparse.ArgumentParser(description="Split a dataset into train/validation.")
    parser.add_argument("dataset", help="path to the csv dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="fraction of data kept for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for a reproducible split (default: 42)")
    args = parser.parse_args()

    train, valid = split_data(args.dataset, args.val_ratio, args.seed)
    train.to_csv("data_train.csv", header=False, index=False)
    valid.to_csv("data_valid.csv", header=False, index=False)
    print(f"train: {len(train)} samples -> data_train.csv")
    print(f"valid: {len(valid)} samples -> data_valid.csv")


if __name__ == "__main__":
    main()
