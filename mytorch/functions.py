import numpy as np
from mytorch.core import Variable, Function, as_array, as_variable


"""
Exp, Sin, Cos
"""


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, dy):
        (x,) = self.inputs
        dx = np.exp(x) * dy
        return dx


def exp(x):
    x = as_array(x)
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, dy):
        (x,) = self.inputs
        dx = dy * cos(x)
        return dx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, dy):
        (x,) = self.inputs
        dx = -dy * sin(x)
        return dx


def cos(x):
    return Cos()(x)


"""
Activation functions
"""


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = dy * (1 - y**2)
        return dx


def tanh(x):
    return Tanh()(x)


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = dy * y * (1 - y)
        return dx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(0, x)
        return y

    def backward(self, dy):
        (x,) = self.inputs
        self.mask = (x > 0).astype(float)
        dx = dy * self.mask
        return dx


def relu(x):
    return ReLU()(x)


"""
Tensor operations : Reshape, Transpose, Sum
"""


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, dy):
        return reshape(dy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)  # No need to reshape
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = np.transpose(x, self.axes)
        return y

    def backward(self, dy):
        if self.axes is None:
            return transpose(dy)

        inv_axes = tuple(np.argsort(list(self.axes)))
        return transpose(dy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)
