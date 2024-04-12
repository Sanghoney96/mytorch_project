if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mytorch.functions as F
from mytorch import Variable


# x0 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.sum(x0, axis=0)
# y.backward()
# print(y)
# print(x0.grad)

# x1 = Variable(np.random.randn(2, 3, 4, 5))
# y = x1.sum(axis=(1, 2), keepdims=True)
# print(y.shape)
# y.backward()
# print(x1.grad.shape)

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)
