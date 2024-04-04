if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch import Variable


def sphere(x, y):
    return x**2 + y**2


def rosenbrock(x, y):
    z = 100 * (y - x**2) ** 2 + (1 - x) ** 2
    return z


def matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y


def goldstein(x, y):
    return (
        1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()

print(x.grad, y.grad)
