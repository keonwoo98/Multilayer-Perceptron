"""A single fully-connected (dense) layer of the network."""

import numpy as np

from srcs.activations import sigmoid, sigmoid_derivative, softmax


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
        """Compute the layer output: activation(W @ x + b).

        The input is cached because backward() needs it to build the
        weight gradient.
        """
        self.input = x
        z = self.W @ x + self.b
        if self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "softmax":
            return softmax(z)
        else:
            raise ValueError(f"unknown activation: {self.activation}")

    def backward(self, delta):
        """Given delta = dLoss/dz for this layer, store the gradients and
        return dLoss/dz for the previous layer.

        delta:  error at this layer's pre-activation z, shape (n_neurons,)
        returns: error to pass to the previous layer, shape (n_inputs,)
        """
        self.dW = np.outer(delta, self.input)
        self.db = delta
        # Propagate to the previous layer: (W^T @ delta) is dLoss/d(prev output),
        # then multiply by the previous layer's sigmoid derivative.
        # This layer's input IS the previous layer's output, so reuse it.
        da_prev = self.W.T @ delta
        return da_prev * sigmoid_derivative(self.input)

    def update(self, lr):
        """Apply one gradient-descent step: W -= lr * dW, b -= lr * db."""
        self.W -= lr * self.dW
        self.b -= lr * self.db
