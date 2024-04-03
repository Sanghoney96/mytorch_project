import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from core import Variable


def numerical_grad(func, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y1.data - y0.data) / (2 * eps)
