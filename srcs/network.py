"""The full network: a stack of dense layers."""

import numpy as np

from srcs.layer import DenseLayer


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

    def update(self, lr):
        """Apply one gradient-descent step on every layer."""
        for layer in self.layers:
            layer.update(lr)

    def predict(self, x):
        """Alias for forward, used at prediction time."""
        return self.forward(x)
