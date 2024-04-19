import numpy as np
from mytorch.core import Variable, Function, as_array, as_variable
from mytorch.utils import sum_to, reshape_sum_backward


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
        mask = (x.data > 0).astype(float)
        dx = dy * mask
        return dx


def relu(x):
    return ReLU()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        x_max = np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        y = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = y * (dy - sum(y * dy, axis=self.axis, keepdims=True))

        return dx


def softmax(x, axis=1):
    return Softmax(axis)(x)


"""
Tensor operations : Reshape, Transpose, BroadcastTo, SumTo, Sum, Matmul
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


class Squeeze(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        y = np.squeeze(x, axis=self.axis)
        return y

    def backward(self, dy):
        dx = expand_dims(dy, self.axis)
        return dx


def squeeze(x, axis):
    return Squeeze(axis)(x)


class ExpandDims(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        y = np.expand_dims(x, axis=self.axis)
        return y

    def backward(self, dy):
        return squeeze(dy, self.axis)


def expand_dims(x, axis):
    return ExpandDims(axis)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, dy):
        (x,) = self.inputs
        return GetItemGrad(self.slices, x.shape)(dy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, dy):
        dx = np.zeros(self.in_shape)
        np.add.at(dx, self.slices, dy)
        return dx

    def backward(self, ddx):
        return get_item(ddx, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, dy):
        dx = sum_to(dy, self.x_shape)
        return dx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = sum_to(x, self.shape)
        return y

    def backward(self, dy):
        dx = broadcast_to(dy, self.x_shape)
        return dx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, dy):
        dy = reshape_sum_backward(dy, self.x_shape, self.axis, self.keepdims)
        dx = broadcast_to(dy, self.x_shape)
        return dx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class Matmul(Function):
    def forward(self, x, W):
        y = np.dot(x, W)
        return y

    def backward(self, dy):
        x, W = self.inputs
        dx = matmul(dy, W.T)
        dW = matmul(x.T, dy)
        return dx, dW


def matmul(x, W):
    return Matmul()(x, W)


"""
Loss functions: MSE, BCE(Binary Crossentropy), CCE(Categorical Crossentropy)
"""


class MSE(Function):
    def forward(self, y, y_hat):
        diff = y - y_hat
        j = np.sum(diff**2) / diff.shape[0]
        return j

    def backward(self, dj):
        y, y_hat = self.inputs
        dy = dj * 2 * (y - y_hat) / y.shape[0]
        dy_hat = -dy
        return dy, dy_hat


def mse(y, y_hat):
    return MSE()(y, y_hat)


class CrossEntropyLoss(Function):
    def forward(self, l, t):
        N = l.shape[0]

        # softmax
        l_max = np.max(l, axis=1, keepdims=True)
        exp_l = np.exp(l - l_max)
        p = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        # cross entropy
        log_p = np.log(p + 1e-7)
        log_p = log_p[np.arange(N), t.ravel()]
        j = -np.sum(log_p) / N
        return j

    def backward(self, dj):
        l, t = self.inputs
        N, K = l.shape

        p = softmax(l)
        t_onehot = np.eye(K, dtype=t.dtype)[t.data]  # convert to one-hot
        dl = (p - t_onehot) * dj / N
        return dl


def crossentropy_loss(l, t):
    return CrossEntropyLoss()(l, t)


"""
Layers
"""


class Linear(Function):
    def forward(self, x, W, b):
        y = np.dot(x, W)
        if b is not None:
            y += b
        return y

    def backward(self, dy):
        x, W, b = self.inputs
        dx = matmul(dy, W.T)
        dW = matmul(x.T, dy)
        if b.data is None:
            db = None
        else:
            db = sum(dy, axis=0)

        return dx, dW, db


def linear(x, W, b):
    return Linear()(x, W, b)
