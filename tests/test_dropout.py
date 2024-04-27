if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from mytorch import test_mode, Variable
import mytorch.functions as F

x = Variable(np.random.rand(5))
print(x)

y = F.dropout(x, p=0.8)
print(y)
y.backward()
print(x.grad)

with test_mode():
    y = F.dropout(x)
    print(y)
