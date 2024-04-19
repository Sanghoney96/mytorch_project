if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mytorch.functions as F
from mytorch import Variable, Parameter
from mytorch.models import MLP


x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [-2.2, 0.5]])
t = np.array([2, 0, 1, 0])
y = MLP((10, 3))(x)

loss = F.crossentropy_loss(y, t)
print(loss)
