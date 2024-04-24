if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from mytorch.dataset import CIFAR10, MNIST


train_set = MNIST(train=True)
test_set = MNIST(train=False)

print(len(train_set))
print(len(test_set))

x, t = train_set[1121]
print(x.shape)

# plt.imshow(x.reshape(28, 28), cmap="gray")
# plt.axis("off")
# plt.show()
# print(x.shape)
