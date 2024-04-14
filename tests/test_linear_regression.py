if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mytorch.functions as F
from mytorch import Variable


x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

W = Variable(np.random.rand(1, 1))
b = Variable(np.random.rand(1))


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = F.linear(x, W, b)
    loss = F.mse(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward(create_graph=False)

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 9:
        print(W, b, loss.data)
