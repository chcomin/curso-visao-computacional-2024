import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transf

class Subset(Dataset):
    """Cria um novo dataset a partir de algumas imagens de um dataset
    de entrada."""

    def __init__(self, ds, indices, transform=None):
        """
        Args:
            ds: dataset original
            indices: lista de índices a serem utilizados no dataset original
            transform: função de transformação dos dados
        """

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

class OxfordIIITPet(Dataset):
    """Dataset com imagens da base Oxford Pets."""

    def __init__(self, root, transforms=None):

        root = Path(root)
        images_folder = root / "images"
        anns_file = root / "annotations/list.txt"

        images = []
        labels = []
        for line in open(anns_file).read().splitlines():
            if line[0]!="#":   # Remove comentários do arquivo
                name, class_id, species_id, breed_id = line.strip().split()
                images.append(images_folder/f'{name}.jpg')
                # -1 para começar em 0
                labels.append(int(species_id)-1)

        self.classes = ['Cat', 'Dog']
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx, apply_transform=True):

        # .convert("RGB") para garantir que as imagens são coloridas
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.labels[idx]

        if self.transforms and apply_transform:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.images)

class TransformsTrain:
    """Transformações de treinamento"""

    def __init__(self, resize_size=224):

        transforms = transf.Compose([
            transf.PILToTensor(),
            transf.RandomResizedCrop(size=(resize_size,resize_size), scale=(0.5,1.),
                                     ratio=(0.7,1.3), antialias=True),
            transf.RandomHorizontalFlip(),
            transf.ToDtype(torch.float32),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(img)

class TransformsEval:
    """Transformações de validação"""

    def __init__(self):

        transforms = transf.Compose([
            transf.PILToTensor(),
            # Por questões históricas (artigo AlexNet), é comum redimensionar
            # a imagem para o tamanho 256x256 e depois aplicar um crop de tamanho
            # 224x224
            transf.Resize(size=256, antialias=True),
            transf.CenterCrop(size=224),
            transf.ToDtype(torch.float32),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(img)

def unormalize(img):
    """Reverte as transformações para visualização da imagem."""

    img = img.permute(1, 2, 0)
    mean = torch.tensor([122.7, 114.6, 100.9])
    std = torch.tensor([59.2, 58.4, 59.0])
    img = img*std + mean
    img = img.to(torch.uint8)

    return img

def get_dataset(root, split=0.2, resize_size=224):
    """Retorna o dataset Oxford Pets.

    Args:
        root: diretório raíz do dataset
        split: fração de dados a utilizar no conjunto de validação
        resize_size: tamanho que as imagens serão redimensionadas
    """

    # Ponderação das classes calculada em um dos notebooks
    class_weights = (0.677, 0.323)

    ds = OxfordIIITPet(root)
    n = len(ds)
    n_valid = int(n*split)

    # Cria uma lista aleatória de índices
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval())

    return ds_train, ds_valid, class_weights
