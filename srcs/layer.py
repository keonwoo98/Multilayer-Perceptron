"""A single fully-connected (dense) layer of the network."""

import numpy as np

from srcs.activations import sigmoid, softmax


class DenseLayer:
    """One layer: holds weights and a bias, and computes its forward pass.

    A layer with `n_neurons` neurons, each receiving `n_inputs` inputs,
    stores its weights in a matrix W of shape (n_neurons, n_inputs).
    """

    def __init__(self, n_inputs, n_neurons, activation="sigmoid", seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Small random weights break symmetry between neurons.
        self.W = np.random.randn(n_neurons, n_inputs) * 0.01
        self.b = np.zeros(n_neurons)
        self.activation = activation

    def forward(self, x):
        """Compute the layer output: activation(W @ x + b)."""
        z = self.W @ x + self.b
        if self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "softmax":
            return softmax(z)
        else:
            raise ValueError(f"unknown activation: {self.activation}")
