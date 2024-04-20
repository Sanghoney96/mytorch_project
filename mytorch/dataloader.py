import math
import random
import numpy as np
from mytorch.dataset import Dataset


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        assert isinstance(dataset, Dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_iter = math.ceil(len(dataset) / batch_size)

        self.reset_iter()

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
        x = np.array([example[0] for example in batch])
        y = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, y

    def next(self):
        return self.__next__()
