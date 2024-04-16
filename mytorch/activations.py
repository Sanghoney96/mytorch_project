import numpy as np


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
