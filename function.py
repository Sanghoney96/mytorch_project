import numpy as np
from variable import Variable
from utils import as_array
import weakref


"""
Function : Base class of functions
"""


class Function:
    def __call__(self, *inputs):
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
Implemented functions
"""


class Multiply(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, dy):
        x0, x1 = self.inputs[0], self.inputs[1]
        dx0 = dy * x0
        dx1 = dy * x1
        return dx0, dx1


def multiply(x0, x1):
    return Multiply()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, dy):
        x = self.inputs[0].data
        dx = 2 * x * dy
        return dx


def square(x):
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
    return Exp()(x)


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


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, dy):
        dx0 = dy
        dx1 = dy
        return dx0, dx1


def add(x0, x1):
    return Add()(x0, x1)


# test backward

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
y.backward()

print(y.grad)
print(x0.grad)

with no_grad():
    y = add(x0, x1)
    y.backward()
