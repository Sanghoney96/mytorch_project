if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mytorch.functions as F
from mytorch import Variable


x = Variable(np.random.randn(4, 2, 3, 5))
y = F.transpose(x, axes=(1, 0, 2, 3))
y.backward(retain_grad=True)
print(y.shape)
print(x.grad.shape)
