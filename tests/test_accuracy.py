if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch.utils import accuracy


x = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
y = np.array([1, 2, 0])

acc = accuracy(x, y)
print(acc)  # 0.6666666666666666
