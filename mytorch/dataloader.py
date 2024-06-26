import math

pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import numpy as np
from mytorch.dataset import Dataset
from mytorch import cuda


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        assert isinstance(dataset, Dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_iter = math.ceil(len(dataset) / batch_size)
        self.gpu = gpu

        self.reset_iter()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True

    def reset_iter(self):
        self.iteration = 0
        if self.shuffle:
            self.idx = np.random.permutation(len(self.dataset))
        else:
            self.idx = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset_iter()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.idx[i * batch_size : (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        y = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, y

    def next(self):
        return self.__next__()
