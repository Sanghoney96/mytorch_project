import numpy as np


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []  # preprocessing functions (ex : weight decay, ...)

    # Set model or layer as target
    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.step(param)

    def step(self, param):
        raise NotImplementedError

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def step(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD:
    def __init__(self, lr, momentum=0.9, weight_decay=0):
        self.lr = lr
        self.momentum = momentum
        self.decay = weight_decay
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            if self.decay != 0:
                grads[key] += self.decay * params[key]

            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr, weight_decay=0):
        self.lr = lr
        self.decay = weight_decay
        self.h = None

    def step(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            if self.decay != 0:
                grads[key] += self.decay * params[key]

            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSProp:
    def __init__(self, lr, alpha=0.99, weight_decay=0):
        self.lr = lr
        self.alpha = alpha
        self.decay = weight_decay
        self.h = None

    def step(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            if self.decay != 0:
                grads[key] += self.decay * params[key]

            self.h[key] = self.alpha * self.h[key] + (1 - self.alpha) * grads[key] ** 2
            params[key] -= (self.lr / np.sqrt(self.h[key]) + 1e-7) * grads[key]


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, weight_decay=0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.t += 1

        for key in params.keys():
            if self.decay != 0:
                grads[key] += self.decay * params[key]

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            params[key] -= (self.lr / (np.sqrt(v_hat) + 1e-7)) * m_hat
