import random
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms.v2 as transforms_pt

class Subset(Dataset):

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

def augment_mnist(img, mean=33., std=76.5):

    color = transforms_pt.ColorJitter(brightness=0.9)
    crop = transforms_pt.RandomResizedCrop(size=28, scale=(0.9, 1.1), ratio= (0.8, 1.2))
    #flip = transforms_pt.RandomHorizontalFlip(p=0.5)
    img = color(img)
    img = transform_mnist(img, mean=mean, std=std)

    return crop(img)

def load_mnist(root='../data', n=1000):

    ds = datasets.MNIST(root, train=True, download=True)
    random.seed(42)
    indices = random.sample(range(len(ds)), k=2*n)
    ds_train = Subset(ds, indices[:n], transform_mnist)
    ds_valid = Subset(ds, indices[n:], transform_mnist)

    return ds_train, ds_valid

def load_mnist_small(root='../data', n_train=50, n_valid=1000):

    ds = datasets.MNIST(root, train=True, download=True)
    train_indices, valid_indices = small_split(ds, n_train=n_train, n_valid=n_valid)

    ds_train = Subset(ds, train_indices, transform_mnist)
    ds_valid = Subset(ds, valid_indices, transform_mnist)

    return ds_train, ds_valid

def load_fashion_mnist(root='../data', n=1000):

    ds = datasets.FashionMNIST(root, train=True, download=True)
    random.seed(42)
    indices = random.sample(range(len(ds)), k=2*n)

    transform = partial(transform_mnist, mean=73., std=81.7)
    ds_train = Subset(ds, indices[:n], transform)
    ds_valid = Subset(ds, indices[n:], transform)

    return ds_train, ds_valid

def load_fashion_mnist_small(root='../data', n_train=50, n_valid=1000):

    ds = datasets.FashionMNIST(root, train=True, download=True)
    train_indices, valid_indices = small_split(ds, n_train=n_train, n_valid=n_valid)

    transform = partial(transform_mnist, mean=73., std=81.7)
    ds_train = Subset(ds, train_indices, transform)
    ds_valid = Subset(ds, valid_indices, transform)

    return ds_train, ds_valid

def small_split(ds, n_train=50, n_valid=1000):

    nc = n_train//10  # Images per class
    train_indices = []
    counts = {c:0 for c in range(10)}
    idx = 0
    while len(train_indices)<n_train:
        _, target = ds[idx]
        if counts[target]<nc:
            counts[target] += 1
            train_indices.append(idx)
        idx += 1
    # Índices restantes
    indices = list(set(range(len(ds))) - set(train_indices))
    random.seed(42)
    # Amostra n_valid índices restantes
    valid_indices = random.sample(indices, k=n_valid)

    return train_indices, valid_indices