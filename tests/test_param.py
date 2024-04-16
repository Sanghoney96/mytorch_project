if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mytorch.functions as F
from mytorch import Variable, Parameter
from mytorch.layers import Layer

layer = Layer()

layer.p1 = Parameter(np.array(1.0))
layer.p2 = Parameter(np.array(2.0))
layer.p3 = Variable(np.array(3.0))
layer.p4 = "test"

print(layer._params)
for name in layer._params:
    print(name, layer.__dict__[name])
