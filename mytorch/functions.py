import numpy as np
import mytorch
from mytorch.core import Variable, Function, as_array, as_variable
from mytorch import cuda, utils


"""
Exp, Sin, Cos
"""


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = y * dy
        return dx


def exp(x):
    x = as_array(x)
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, dy):
        (x,) = self.inputs
        dx = dy * cos(x)
        return dx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
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
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = dy * (1 - y**2)
        return dx


def tanh(x):
    return Tanh()(x)


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = 1 / (1 + xp.exp(-x))
        return y

    def backward(self, dy):
        y = self.outputs[0]()
        dx = dy * y * (1 - y)
        return dx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(0, x)
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
        xp = cuda.get_array_module(x)
        x_max = xp.max(x, axis=self.axis, keepdims=True)
        exp_x = xp.exp(x - x_max)
        y = exp_x / xp.sum(exp_x, axis=self.axis, keepdims=True)
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
        y = x.transpose(self.axes)
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
        y = x.squeeze(axis=self.axis)
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
        y = x.expand_dims(axis=self.axis)
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
        xp = cuda.get_array_module(dy)
        dx = xp.zeros(self.in_shape)

        if xp is np:
            np.add.at(dx, self.slices, dy)
        else:
            xp.scatter_add(dx, self.slices, dy)
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
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
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
        y = utils.sum_to(x, self.shape)
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
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, dy):
        dy = utils.reshape_sum_backward(dy, self.x_shape, self.axis, self.keepdims)
        dx = broadcast_to(dy, self.x_shape)
        return dx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class Matmul(Function):
    def forward(self, x, W):
        y = x.dot(W)
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
        j = (diff**2).sum() / diff.shape[0]
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
        xp = cuda.get_array_module(l)

        # softmax
        l_max = xp.max(l, axis=1, keepdims=True)
        exp_l = xp.exp(l - l_max)
        p = exp_l / xp.sum(exp_l, axis=1, keepdims=True)

        # cross entropy
        log_p = xp.log(p + 1e-7)
        log_p = log_p[xp.arange(N), t.ravel()]
        j = -xp.sum(log_p) / N
        return j

    def backward(self, dj):
        l, t = self.inputs
        N, K = l.shape

        p = softmax(l)

        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(K, dtype=t.dtype)[t.data]  # convert to one-hot
        dl = (p - t_onehot) * dj / N
        return dl


def crossentropy_loss(l, t):
    return CrossEntropyLoss()(l, t)


"""
Layers
"""


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
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


class Dropout(Function):
    def __init__(self, p=0.5):
        self.dropout_ratio = p

    def forward(self, x):
        if mytorch.Config.train:
            xp = cuda.get_array_module(x)
            mask = xp.random.rand(*x.shape) > self.dropout_ratio
            scale = 1 - self.dropout_ratio
            y = x * mask / scale
            return y
        else:
            return x

    def backward(self, dy):
        if mytorch.Config.train:
            y = self.outputs[0]()
            mask = (y.data != 0).astype(float)
            scale = 1 - self.dropout_ratio
            dx = dy * mask / scale
            return dx
        else:
            return dy


def dropout(x, p=0.5):
    return Dropout(p)(x)
