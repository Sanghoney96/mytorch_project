import numpy as np

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


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(len(x))
print(x.shape)
print(x)
