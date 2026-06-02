"""Activation functions and their derivatives."""

import numpy as np


def sigmoid(z):
    """Squash any value into the range (0, 1): 1 / (1 + e^-z)."""
    # Clip z to avoid overflow in exp() for extreme inputs.
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid, written with its own output: a * (1 - a).

    `a` is the value already returned by sigmoid(z), not the raw z.
    """
    return a * (1 - a)


def softmax(z):
    """Turn a score vector into a probability distribution that sums to 1.

    Subtract the max before exp() to prevent overflow. This does not
    change the result, since numerator and denominator shrink equally.
    """
    z_shift = z - np.max(z)
    exp = np.exp(z_shift)
    return exp / np.sum(exp)
