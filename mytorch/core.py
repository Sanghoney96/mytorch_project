import numpy as np
import weakref
from contextlib import contextmanager


"""
Variable : Contain input data.
"""


class Variable:
    def __init__(self, data, name=None):
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError(f"{type(data)} is not supported")

        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    __array_priority__ = 1972  # to call operator of Variable first

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __neg__(self):
        return neg(self)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                seen_set.add(f)  # for avoiding duplication of functions
                funcs.append(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)  # sort functions by generation

        while funcs:
            func = funcs.pop()

            # backpropagation
            dys = [y().grad for y in func.outputs]

            with using_config("enable_backprop", create_graph):
                dxs = func.backward(*dys)

                if not isinstance(dxs, tuple):
                    dxs = (dxs,)

                for x, dx in zip(func.inputs, dxs):
                    if x.grad is None:
                        x.grad = dx
                    else:
                        x.grad = x.grad + dx  # Add gradient if x.grad already exists.

                    if x.creator:
                        add_func(x.creator)

            if not retain_grad:
                for y in func.outputs:
                    y().grad = None


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    elif isinstance(obj, np.ndarray):
        return Variable(obj)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


"""
Function : Base class of functions
"""


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        # forwardpropagation
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            # reference to the creator, inputs, and outputs
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *dys):
        raise NotImplementedError()


"""
Config
"""


class Config:
    enable_backprop = True


@contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


"""
Operations : Add, Mul, Neg, Sub, Div, Square, Pow
"""


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, dy):
        dx0 = dy
        dx1 = dy
        return dx0, dx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, dy):
        x0, x1 = self.inputs
        dx0 = dy * x1
        dx1 = dy * x0
        return dx0, dx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, dy):
        return -dy


def neg(x):
    x = as_array(x)
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, dy):
        dx0 = dy
        dx1 = -dy
        return dx0, dx1


def sub(x0, x1):
    x0, x1 = as_array(x0), as_array(x1)
    return Sub()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, dy):
        x0, x1 = self.inputs
        dx0 = dy / x1
        dx1 = -dy * (x0 / x1**2)
        return dx0, dx1


def div(x0, x1):
    x0, x1 = as_array(x0), as_array(x1)
    return Div()(x0, x1)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, dy):
        (x,) = self.inputs
        c = self.c
        dx = c * x ** (c - 1) * dy
        return dx


def pow(x, c):
    return Pow(c)(x)


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, dy):
        (x,) = self.inputs
        dx = 2 * x * dy
        return dx


def square(x):
    x = as_array(x)
    return Square()(x)


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


"""
Sin / Cos
"""


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
