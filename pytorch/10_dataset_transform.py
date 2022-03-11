import torch
import torchvision #datasets, models for computer vision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('pytorch/data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)

        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

if __name__ == "__main__":
    dataset = WineDataset(transform=None)
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features)
    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
    dataset = WineDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data
    print(features)
    print(type(features), type(labels))