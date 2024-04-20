import numpy as np
from mytorch.core import as_variable


def sum_to(x, shape):
    lead = x.ndim - len(shape)
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    tupled_axis = axis

    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def accuracy(y_pred, y):
    y_pred, y = as_variable(y_pred), as_variable(y)

    pred = y_pred.data.argmax(axis=1).reshape(y.shape)
    acc = (pred == y.data).mean()
    return acc
