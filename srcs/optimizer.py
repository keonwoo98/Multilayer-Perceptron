"""Optimizers: the rules that turn gradients into weight updates.

Each optimizer exposes step(layers, grads), where grads is a list of
(dW, db) tuples already averaged over the mini-batch, aligned with layers.
"""

import numpy as np


class SGD:
    """Stochastic gradient descent with optional L2 weight decay.

    Weight decay adds `weight_decay * W` to the gradient, which pulls
    weights toward zero and reduces over-confident predictions.
    """

    def __init__(self, lr=0.1, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers, grads):
        for layer, (dW, db) in zip(layers, grads):
            layer.W -= self.lr * (dW + self.weight_decay * layer.W)
            layer.b -= self.lr * db


class Adam:
    """Adam optimizer (adaptive moment estimation).

    Keeps a running average of past gradients (m) and squared gradients (v)
    per parameter, giving each weight its own adaptive step size. Usually
    converges faster and more smoothly than plain SGD.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.state = {}

    def step(self, layers, grads):
        self.t += 1
        for idx, (layer, (dW, db)) in enumerate(zip(layers, grads)):
            if idx not in self.state:
                self.state[idx] = {
                    "mW": np.zeros_like(layer.W), "vW": np.zeros_like(layer.W),
                    "mb": np.zeros_like(layer.b), "vb": np.zeros_like(layer.b),
                }
            st = self.state[idx]
            if self.weight_decay:
                dW = dW + self.weight_decay * layer.W
            layer.W -= self._adam_step(dW, st, "mW", "vW")
            layer.b -= self._adam_step(db, st, "mb", "vb")

    def _adam_step(self, grad, st, m_key, v_key):
        """Compute one Adam update term for a single parameter array."""
        st[m_key] = self.beta1 * st[m_key] + (1 - self.beta1) * grad
        st[v_key] = self.beta2 * st[v_key] + (1 - self.beta2) * (grad ** 2)
        m_hat = st[m_key] / (1 - self.beta1 ** self.t)
        v_hat = st[v_key] / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
