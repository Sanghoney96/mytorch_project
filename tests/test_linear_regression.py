if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mytorch.functions as F
import mytorch.layers as L
from mytorch.models import MLP
from mytorch.optimizers import SGD, MomentumSGD

x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
max_iter = 10000

model = MLP((10, 10, 1))
optimizer = MomentumSGD(lr)
optimizer.setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mse(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)
