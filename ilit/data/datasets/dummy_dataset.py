from .dataset import dataset_registry, Dataset

@dataset_registry(dataset_type="dummy", framework="tensorflow, pytorch, mxnet", dataset_format='')
class DummyDataset(Dataset):
    """Dataset used for dummy data generation.
       This Dataset is to construct a dataset from a specific shape.
       (TODO) construct dummy data from real dataset or iteration of data.

    """
    def __init__(self, shape, transform=None):
        self.transform = transform
        shape = tuple(shape)
        import numpy as np
        self.dataset = np.random.random(size=shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.transform != None:
            sample = self.transform(sample)
        return sample

