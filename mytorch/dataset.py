import numpy as np


class Dataset:
    def __init__(self, train=True, transform=None, label_transform=None):
        self.train = train
        self.transform = transform
        self.label_transform = label_transform

        if self.transform is None:
            self.transform = lambda x: x
        if self.label_transform is None:
            self.label_transform = lambda x: x

        self.data = None
        self.label = None
        self.get_data()

    def __getitem__(self, idx):
        assert np.isscalar(idx)

        if self.label is None:
            return self.transform(self.data[idx]), None
        else:
            return self.transform(self.data[idx]), self.label_transform(self.label[idx])

    def __len__(self):
        return len(self.data)

    def get_data(self):
        pass
