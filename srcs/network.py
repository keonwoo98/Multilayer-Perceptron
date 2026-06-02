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

    def predict(self, x):
        """Alias for forward, used at prediction time."""
        return self.forward(x)
