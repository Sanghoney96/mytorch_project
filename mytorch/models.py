import numpy as np
import mytorch.layers as L
import mytorch.functions as F

from mytorch import Layer, utils


"""
Model class
"""


class Model(Layer):
    pass


class MLP(Model):
    def __init__(self, out_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(out_sizes):
            layer = L.Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
