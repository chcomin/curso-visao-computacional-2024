import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms.v2 as transforms_pt

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

class Transform:
    """Define transformações a serem aplicadas em um conjunto de imagens."""

    def __init__(self, mean, std, augment=False):

        self.mean = mean
        self.std = std
        self.augment = augment

        if augment:
            # Altera aleatoriamente o brilho das imagens
            self.color = transforms_pt.ColorJitter(brightness=0.9)
            # Recorta aleatoriamente uma parte da imagem e redimensiona o resultado
            # para o tamanho 28x28
            self.crop = transforms_pt.RandomResizedCrop(size=28, scale=(0.9, 1.1), ratio= (0.8, 1.2))
        else:
            self.color = None
            self.crop = None

    def normalize(self, img):
        # Conversão PIL->numpy
        img = np.array(img, dtype=np.float32)
        # Conversão numpy->pytorch
        img = torch.from_numpy(img)
        # Normalização
        img = (img-self.mean)/self.std
        # Adição de um canal
        img = img.reshape(1, img.shape[0], img.shape[1])

        return img

    def __call__(self, img):

        if self.augment:
            img = self.color(img)
        img = self.normalize(img)
        if self.augment:
            img = self.crop(img)

        return img

def load_fashion_mnist_small(root='../data', n_train=50, n_valid=1000, augment=False):
    '''Seleciona `n_train`/10 imagens de cada classe do Fashion MNIST e cria um
    dataset.'''

    ds = datasets.FashionMNIST(root, train=True, download=True)
    train_indices, valid_indices = small_split(ds, n_train=n_train, n_valid=n_valid)

    transform = Transform(mean=73., std=81.7, augment=augment)
    ds_train = Subset(ds, train_indices, transform)
    ds_valid = Subset(ds, valid_indices, transform)

    return ds_train, ds_valid

def small_split(ds, n_train=50, n_valid=1000):
    '''Percorre um dataset e seleciona as primeiras `n_train`/10 imagens
    encontradas de cada classe. As imagens não selecionadas são incluídas
    em um dataset de validação. Assume que o datast possui 10 classes.'''

    nc = n_train//10  # Imagens por classe
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
