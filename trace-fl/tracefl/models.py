"""One-hidden-layer ReLU MLP with a flat parameter vector (NumPy only)."""
import numpy as np


class MLP:
    """MLP d -> h -> c. Parameters live in one flat float32 vector, which is
    what clients send and what the aggregators operate on."""

    def __init__(self, d, h, c):
        self.d, self.h, self.c = d, h, c
        self.n = d * h + h + h * c + c

    def init(self, rng):
        w = np.zeros(self.n, np.float32)
        i = 0
        w[i:i + self.d * self.h] = rng.normal(0, np.sqrt(2 / self.d), self.d * self.h)
        i += self.d * self.h + self.h
        w[i:i + self.h * self.c] = rng.normal(0, np.sqrt(1 / self.h), self.h * self.c)
        return w

    def unpack(self, w):
        i = 0
        W1 = w[i:i + self.d * self.h].reshape(self.d, self.h); i += self.d * self.h
        b1 = w[i:i + self.h]; i += self.h
        W2 = w[i:i + self.h * self.c].reshape(self.h, self.c); i += self.h * self.c
        return W1, b1, W2, w[i:]

    def logits(self, w, X):
        W1, b1, W2, b2 = self.unpack(w)
        return np.maximum(X @ W1 + b1, 0) @ W2 + b2

    def grad(self, w, X, y):
        """Gradient of the mean softmax cross-entropy."""
        W1, b1, W2, b2 = self.unpack(w)
        H = np.maximum(X @ W1 + b1, 0)
        Z = H @ W2 + b2
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        P[np.arange(len(y)), y] -= 1
        P /= len(y)
        gW2 = H.T @ P
        gb2 = P.sum(0)
        dH = (P @ W2.T) * (H > 0)
        return np.concatenate([(X.T @ dH).ravel(), dH.sum(0), gW2.ravel(), gb2]).astype(np.float32)

    def accuracy(self, w, X, y):
        return float((self.logits(w, X).argmax(1) == y).mean())


def local_update(model, w, X, y, rng, lr, epochs, batch_size):
    """Local mini-batch SGD; returns the model delta u = w_local - w."""
    v = w.copy()
    for _ in range(epochs):
        p = rng.permutation(len(y))
        for b in range(0, len(y), batch_size):
            j = p[b:b + batch_size]
            v -= lr * model.grad(v, X[j], y[j])
    return v - w
