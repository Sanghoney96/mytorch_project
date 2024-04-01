import numpy as np


class Linear:
    def __init__(self, features_in, features_out, weight_initializer="glorot"):
        if weight_initializer == "glorot":
            self.W = np.random.randn(features_in, features_out) * np.sqrt(
                1 / features_in
            )
        elif weight_initializer == "he":
            self.W = np.random.randn(features_in, features_out) * np.sqrt(
                2 / features_in
            )
        self.b = np.zeros((1, features_out))

        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        Z = np.dot(self.X, self.W) + self.b

        return Z

    def backward(self, dZ):
        dX = np.dot(dZ, self.W.T)
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0)

        return dX


class Dropout:
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.mask = None

    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.ratio).astype(float)
            return X * self.mask
        else:
            return X * (1 - self.ratio)

    def backward(self, dX):
        return dX * self.mask


# Dropout test
X = np.random.randn(2, 5)
dropout = Dropout()
print("X:")
print(X)
print("Forward:")
print(dropout.forward(X))
print("Backward:")
print(dropout.backward(np.ones_like(X)))
