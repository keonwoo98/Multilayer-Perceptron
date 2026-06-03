# Multilayer Perceptron

A multilayer perceptron implemented from scratch (only numpy / pandas for
linear algebra and matplotlib for plotting) to classify breast-cancer cell
nuclei as malignant (M) or benign (B) on the Wisconsin dataset.

feedforward, backpropagation and mini-batch gradient descent are all
implemented by hand.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Three programs: split, train, predict.

### 1. Split the dataset

```bash
python split.py data.csv --val_ratio 0.2 --seed 42
# -> data_train.csv, data_valid.csv
```

### 2. Train

```bash
# with a separate validation file
python train.py data_train.csv --valid data_valid.csv

# or with a single file (validation is split off internally) -- this is
# what works with the evaluation's data_training.csv
python train.py data_training.csv
```

Saves the model (topology + weights + normalization stats) to
`models/model.npz` and displays the learning curves.

Useful options:

| option            | default | meaning                                  |
|-------------------|---------|------------------------------------------|
| `--layers`        | 24 24   | hidden layer sizes                       |
| `--epochs`        | 80      | training epochs                          |
| `--lr`            | 0.3     | learning rate (use ~0.001 for adam)      |
| `--batch_size`    | 16      | mini-batch size                          |
| `--optimizer`     | sgd     | `sgd` or `adam`                          |
| `--weight_decay`  | 1e-3    | L2 regularization strength               |
| `--early_stopping`| off     | stop when validation loss plateaus       |

### 3. Predict

```bash
python predict.py --data data_valid.csv
```

Loads the saved model and reports binary cross-entropy, accuracy,
precision, recall and F1.

## Evaluation workflow

The project evaluation uses `evaluation.py`, which produces
`data_training.csv` (75%) and `data_test.csv` (25%):

```bash
python evaluation.py
python train.py data_training.csv
python predict.py --data data_test.csv
```

## Bonus features

- mini-batch gradient descent
- L2 regularization (weight decay)
- Adam optimizer (`--optimizer adam`)
- early stopping (`--early_stopping`)
- extra metrics: precision, recall, F1

## Project structure

```
split.py / train.py / predict.py   the three programs
srcs/
  activations.py   sigmoid, softmax (+ derivatives)
  preprocessing.py load, standardize, one-hot, train/val split
  layer.py         DenseLayer (forward / backward)
  network.py       Network (feedforward / backprop / fit)
  optimizer.py     SGD, Adam
  losses.py        cross-entropy
  metrics.py       accuracy, precision, recall, F1
  plotting.py      learning curves
docs/              concept notes and design (Korean)
```
