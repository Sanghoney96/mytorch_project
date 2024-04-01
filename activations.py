import numpy as np


class Sigmoid:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        dZ = dA * self.A * (1 - self.A)
        return dZ


class Tanh:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        dZ = dA * (1 - self.A**2)
        return dZ


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, Z):
        self.A = np.maximum(0, Z)
        self.mask = (Z > 0).astype(float)
        return self.A

    def backward(self, dA):
        dZ = dA * self.mask
        return dZ


class Softmax:
    def __init__(self):
        self.Y_hat = None

    def forward(self, Z):
        # Z : (batch_size, features_in)
        # Y_hat : (batch_size, n_classes)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.Y_hat = exp_Z / np.sum(np.exp(Z), axis=1, keepdims=True)
        return self.Y_hat

    def backward(self, dY_hat):
        # dY_hat: (batch_size, n_classes)
        # dY_hat_dZ : (batch_size, n_classes, n_classes)
        Y_hat = self.Y_hat[:, :, np.newaxis]

        Y_hat_diag = Y_hat * np.eye(self.Y_hat.shape[1])
        Y_hat_outer = np.matmul(Y_hat, Y_hat.transpose(0, 2, 1))

        dY_hat_dZ = Y_hat_diag - Y_hat_outer

        dZ = np.matmul(dY_hat[:, np.newaxis, :], dY_hat_dZ)

        return np.squeeze(dZ, axis=1)
