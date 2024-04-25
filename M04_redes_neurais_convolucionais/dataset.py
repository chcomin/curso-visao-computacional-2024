import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

class Subset(Dataset):
    """Cria um subconjunto de um dataset."""

    def __init__(self, ds, indices, transform=None):
        self.ds = ds
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):

        img, target = self.ds[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)

def transform_mnist(img, mean=33., std=76.5):
    # Conversão PIL->numpy
    img = np.array(img, dtype=np.float32)
    # Conversão numpy->pytorch
    img = torch.from_numpy(img)
    # Normalização
    img = (img-mean)/std
    # Adição de um canal
    img = img.reshape(1, img.shape[0], img.shape[1])

    return img

def load_mnist(root='../data', n=1000):

    ds = datasets.MNIST(root, train=True, download=True)
    random.seed(42)
    indices = random.sample(range(len(ds)), k=2*n)
    ds_train = Subset(ds, indices[:n], transform_mnist)
    ds_valid = Subset(ds, indices[n:], transform_mnist)

    return ds_train, ds_valid