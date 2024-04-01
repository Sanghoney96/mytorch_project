import numpy as np

"""
Variable : Contain input data.
"""


class Variable:
    def __init__(self, data):
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def zero_grad(self):
        self.grad = None

    def backward(self):
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
