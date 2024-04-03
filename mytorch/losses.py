import numpy as np


class MSELoss:
    def __call__(self, Y_hat, Y):
        # Y_hat, Y : (batch_size, 1)
        self.Y = Y
        self.Y_hat = Y_hat

        return np.mean((Y - Y_hat) ** 2)

    def backward(self):
        dY_hat = -2 * (self.Y - self.Y_hat) / self.Y.shape[0]
        return dY_hat


class CrossEntropyLoss:
    def __call__(self, Y_hat, Y):
        # Y_hat, Y : (batch_size, n_classes)
        self.Y_hat = Y_hat
        self.Y = Y
        eps = 1e-7

        return -np.mean(np.sum(Y * np.log(Y_hat + eps), axis=1))

    def backward(self):
        dY_hat = -(self.Y / self.Y_hat) / self.Y.shape[0]
        return dY_hat


class BCELoss:
    def __call__(self, Y_hat, Y):
        # Y_hat, Y : (batch_size, 1)
        self.Y = Y
        self.Y_hat = Y_hat
        eps = 1e-7

        return -np.mean(
            self.Y * np.log(self.Y_hat + eps)
            + (1 - self.Y) * np.log(1 - self.Y_hat + eps)
        )

    def backward(self):
        dY_hat = (self.Y_hat - self.Y) / (self.Y_hat * (1 - self.Y_hat))
        dY_hat /= self.Y.shape[0]
        return dY_hat
