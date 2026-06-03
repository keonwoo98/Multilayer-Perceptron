"""The full network: a stack of dense layers."""

import numpy as np

from srcs.layer import DenseLayer
from srcs.losses import categorical_cross_entropy


class Network:
    """A feedforward network built from a list of layer sizes.

    Example: layer_sizes = [30, 24, 24, 2] builds three layers
        30 -> 24 (sigmoid), 24 -> 24 (sigmoid), 24 -> 2 (softmax).
    The last layer uses softmax to output a probability distribution.
    """

    def __init__(self, layer_sizes, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            n_inputs = layer_sizes[i]
            n_neurons = layer_sizes[i + 1]
            is_last = i == len(layer_sizes) - 2
            activation = "softmax" if is_last else "sigmoid"
            self.layers.append(DenseLayer(n_inputs, n_neurons, activation))

    def forward(self, x):
        """Pass the input through every layer in order; return the output."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred, y_true):
        """Propagate the error from the output back to the first layer.

        For softmax output + cross-entropy loss, the output-layer error
        simplifies to (prediction - target). Each layer then stores its
        own gradients and hands the error to the layer before it.
        """
        delta = y_pred - y_true
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def predict(self, x):
        """Alias for forward, used at prediction time."""
        return self.forward(x)

    def evaluate(self, X, y):
        """Compute average loss and accuracy over a dataset.

        X: features (n_samples, n_features); y: one-hot targets (n_samples, 2).
        """
        preds = np.array([self.forward(xi) for xi in X])
        loss = categorical_cross_entropy(preds, y)
        accuracy = np.mean(preds.argmax(axis=1) == y.argmax(axis=1))
        return loss, accuracy

    def fit(self, X_train, y_train, X_val, y_val, epochs, optimizer,
            batch_size=8, early_stopping=False, patience=10):
        """Train with mini-batch gradient descent.

        Each epoch: shuffle, split into mini-batches, and for each batch
        average the per-sample gradients before letting the optimizer
        update the weights. Averaging smooths the updates and curbs the
        over-confident predictions that hurt the cross-entropy loss.

        If early_stopping is on, training stops when the validation loss
        has not improved for `patience` epochs, and the best weights are
        restored.
        """
        history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        n = len(X_train)
        best_val, best_weights, wait = float("inf"), None, 0

        for epoch in range(epochs):
            order = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                grads = [(np.zeros_like(l.W), np.zeros_like(l.b)) for l in self.layers]
                for i in batch:
                    pred = self.forward(X_train[i])
                    self.backward(pred, y_train[i])
                    for k, layer in enumerate(self.layers):
                        grads[k] = (grads[k][0] + layer.dW, grads[k][1] + layer.db)
                m = len(batch)
                grads = [(gW / m, gb / m) for gW, gb in grads]
                optimizer.step(self.layers, grads)

            loss, acc = self.evaluate(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)
            history["loss"].append(loss)
            history["val_loss"].append(val_loss)
            history["acc"].append(acc)
            history["val_acc"].append(val_acc)
            print(f"epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

            if early_stopping:
                if val_loss < best_val:
                    best_val = val_loss
                    best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"> early stopping at epoch {epoch + 1} "
                              f"(best val_loss: {best_val:.4f})")
                        break

        if early_stopping and best_weights is not None:
            for layer, (W, b) in zip(self.layers, best_weights):
                layer.W, layer.b = W, b
        return history
