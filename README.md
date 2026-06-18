# Multilayer Perceptron

A multilayer perceptron implemented from scratch (only numpy / pandas for
linear algebra and matplotlib for plotting) to classify breast-cancer cell
nuclei as malignant (M) or benign (B) on the Wisconsin dataset.

feedforward, backpropagation and mini-batch gradient descent are all
implemented by hand.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate            # or call ./venv/bin/python3 directly
pip install -r requirements.txt
```

Quick check that the only allowed libraries import cleanly:

```bash
./venv/bin/python3 -c "import numpy, pandas, matplotlib; print('ok')"
```

## The three programs

| program      | what it does                                            | output                |
|--------------|---------------------------------------------------------|-----------------------|
| `split.py`   | shuffle and split a dataset into train / validation     | `data_train.csv`, `data_valid.csv` |
| `train.py`   | train the network, save the model, plot learning curves | `models/model.npz`, `learning_curves.png` |
| `predict.py` | load the model and evaluate it (binary cross-entropy)   | metrics printed to stdout |

---

## Evaluation walkthrough (copy-paste, in order)

This is the exact command sequence for a defense. The official performance
score is measured with the evaluator-provided `evaluation.py`.

### 1. Generate the official train/test split

`evaluation.py` is provided by 42 (not in this repo). It shuffles `data.csv`
and writes `data_training.csv` (75%) and `data_test.csv` (25%).

```bash
python3 evaluation.py
```

### 2. Train on the training set

```bash
python3 train.py data_training.csv
```

Prints `loss` / `val_loss` every epoch, saves `models/model.npz`, and shows
the learning curves. The default network is `30 -> 24 -> 24 -> 2`
(two hidden layers), trained with mini-batch SGD + L2 + early stopping.

### 3. Predict on the held-out test set  ← this is the graded score

```bash
python3 predict.py --data data_test.csv
```

Prints binary cross-entropy (the performance score), plus accuracy,
precision, recall and F1. **Always score on `data_test.csv`, never on
`data_training.csv`** — the test set is the only data the model never saw.

> The training is random, so the rules allow training up to 3 times and
> keeping the best test score:
> ```bash
> python3 train.py data_training.csv --seed 1 && python3 predict.py --data data_test.csv
> python3 train.py data_training.csv --seed 7 && python3 predict.py --data data_test.csv
> ```

### 4. Show modularity (change the number of hidden layers)

```bash
python3 train.py data_training.csv --layers 24 24 24 --epochs 40
python3 -c "import numpy as np; print(np.load('models/model.npz')['layer_sizes'])"
# -> [30 24 24 24  2]   (now three hidden layers, no code change)
```

### 5. Inspect what was saved in the model

```bash
python3 -c "import numpy as np; d=np.load('models/model.npz'); print(list(d.keys()))"
# -> ['layer_sizes', 'mean', 'std', 'W0', 'b0', 'W1', 'b1', 'W2', 'b2']
```

The model file holds everything `predict.py` needs to rebuild the exact same
model and transform:

| key          | meaning                                             |
|--------------|-----------------------------------------------------|
| `layer_sizes`| network topology, e.g. `[30 24 24 2]`               |
| `W0,b0,...`  | learned weights and biases for each layer           |
| `mean`,`std` | per-feature normalization stats from the **training** set |

### 6. Show the bonuses

Flags that enable / show each bonus:

| bonus                | flag                                                      |
|----------------------|-----------------------------------------------------------|
| Adam optimizer       | `--optimizer adam --lr 0.001`                             |
| mini-batch GD        | `--batch_size 16` (default; `--batch_size 1` = pure SGD)  |
| L2 regularization    | `--weight_decay 1e-3` (default; `--weight_decay 0` = off) |
| early stopping       | on by default (no flag; `--no_early_stopping` to disable) |
| precision/recall/F1  | always printed by `predict.py`                            |

```bash
# Adam optimizer
python3 train.py data_training.csv --optimizer adam --lr 0.001 --epochs 40

# disable early stopping (e.g. to show overfitting on a long run)
python3 train.py data_training.csv --no_early_stopping --epochs 200
```

precision / recall / F1 are already printed by `predict.py`.

### 7. Error handling (no crash on bad input)

```bash
python3 predict.py --data does_not_exist.csv   # prints an error and exits, no crash
python3 predict.py --data data_test.csv        # model must be trained first
```

---

## Using split.py (own train/validation workflow)

`split.py` is the deliverable for the "Dataset split" item. When you use it,
pass **both** files to `train.py` so the validation set you carved out is the
one actually used:

```bash
python3 split.py data.csv --val_ratio 0.2 --seed 42   # -> data_train.csv, data_valid.csv
python3 train.py data_train.csv --valid data_valid.csv
python3 predict.py --data data_valid.csv
```

If you pass only the training file, `train.py` splits a validation set off
internally instead (this is what makes the single-file `evaluation.py`
workflow above work).

## Command reference (train.py)

| option               | default | meaning                                   |
|----------------------|---------|-------------------------------------------|
| `dataset`            | —       | training csv (positional, required)       |
| `--valid`            | none    | separate validation csv (else split internally) |
| `--layers`           | 24 24   | hidden layer sizes                        |
| `--epochs`           | 80      | max training epochs                       |
| `--lr`               | 0.3     | learning rate (use ~0.001 for adam)       |
| `--batch_size`       | 16      | mini-batch size                           |
| `--optimizer`        | sgd     | `sgd` or `adam`                           |
| `--weight_decay`     | 1e-3    | L2 regularization strength                |
| `--no_early_stopping`| (on)    | disable early stopping (on by default)    |
| `--patience`         | 15      | epochs to wait before early stopping      |
| `--val_ratio`        | 0.2     | validation fraction when no `--valid`     |
| `--seed`             | 42      | reproducible shuffle / weight init        |

## Bonus features

- **Adam optimizer** (`--optimizer adam`) — per-weight adaptive step + momentum
- **early stopping** (on by default; `--no_early_stopping` to disable) — stops
  when validation loss plateaus and restores the best weights
- **mini-batch gradient descent** (`--batch_size`) — averaged gradients per batch
- **L2 regularization / weight decay** (`--weight_decay`) — curbs overconfidence
- **extra metrics** — precision, recall, F1 (printed by `predict.py`)

## Project structure

```
split.py / train.py / predict.py   the three programs
srcs/
  activations.py   sigmoid, softmax (+ derivatives)
  preprocessing.py load, standardize, one-hot, train/val split
  layer.py         DenseLayer (forward / backward)
  network.py       Network (feedforward / backprop / fit)
  optimizer.py     SGD, Adam
  losses.py        cross-entropy (categorical + binary)
  metrics.py       accuracy, precision, recall, F1
  plotting.py      learning curves
docs/디펜스_대본.md  defense script: concepts, commands, Q&A (Korean / English)
```
