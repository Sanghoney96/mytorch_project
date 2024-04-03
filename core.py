import numpy as np
from utils import as_array
import weakref


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

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def zero_grad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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
            xs = func.inputs
            dys = [y().grad for y in func.outputs]
            dxs = func.backward(*dys)

            if not isinstance(dxs, tuple):
                dxs = (dxs,)

            for x, dx in zip(xs, dxs):
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


"""
Function : Base class of functions
"""


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *dys):
        raise NotImplementedError()


class Config:
    enable_backprop = True


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
Implemented operations
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
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        dx0 = dy * x0
        dx1 = dy * x1
        return dx0, dx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, dy):
        x = self.inputs[0].data
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
        x = self.input.data
        dx = np.exp(x) * dy
        return dx


def exp(x):
    x = as_array(x)
    return Exp()(x)


"""
Activation functions
"""


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, dy):
        x = self.input.data
        dx = dy * (1 - x**2)
        return dx


def tanh(x):
    return Tanh()(x)


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, dy):
        x = self.input.data
        dx = dy * x * (1 - x)
        return dx


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(0, x)
        return y

    def backward(self, dy):
        x = self.input.data
        self.mask = (x > 0).astype(float)
        dx = dy * self.mask
        return dx


def relu(x):
    return ReLU()(x)


a = Variable(np.array(3.0))
y = 2 * a + 1
print(y)
