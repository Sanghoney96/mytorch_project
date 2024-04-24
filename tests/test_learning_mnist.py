if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mytorch
import mytorch.functions as F
import mytorch.layers as L
from mytorch.models import MLP
from mytorch.optimizers import SGD, MomentumSGD

from mytorch import DataLoader
from mytorch.dataset import MNIST
from mytorch.utils import accuracy
from mytorch.transforms import Compose, Flatten, Normalize


max_epoch = 5
batch_size = 64
hidden_size = 1000

train_set = MNIST(train=True, transform=Compose([Flatten(), Normalize(0, 255)]))
test_set = MNIST(train=False, transform=Compose([Flatten(), Normalize(0, 255)]))

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = MomentumSGD(lr=0.01, momentum=0.9).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.crossentropy_loss(y, t)
        acc = accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc) * len(t)

    print("epoch: {}".format(epoch + 1))
    print(
        "train loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )

    sum_loss, sum_acc = 0, 0
    with mytorch.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.crossentropy_loss(y, t)
            acc = accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += acc * len(t)

    print(
        "test loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(test_set), sum_acc / len(test_set)
        )
    )
