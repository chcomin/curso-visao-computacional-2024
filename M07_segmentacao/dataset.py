import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transf
from torchvision import tv_tensors

class Subset(Dataset):

    def __init__(self, ds, indices, transform=None):
        self.ds = ds
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):

        img, target = self.ds[self.indices[idx]]
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.indices)

class OxfordIIITPetSeg(Dataset):

    def __init__(self, root, transforms=None, ignore_val=2):

        root = Path(root)
        images_folder = root / "images"
        segs_folder = root / "annotations/trimaps"
        anns_file = root / "annotations/list.txt"

        images = []
        segs = []
        for line in open(anns_file).read().splitlines():
            if line[0]!="#":   # Remove comentários do arquivo
                name, class_id, species_id, breed_id = line.strip().split()
                images.append(images_folder/f'{name}.jpg')
                segs.append(segs_folder/f'{name}.png')

        self.classes = ['Cat', 'Dog']
        self.images = images
        self.segs = segs
        self.transforms = transforms
        self.ignore_val = ignore_val

    def __getitem__(self, idx, apply_transform=True):

        # .convert("RGB") para garantir que as imagens são coloridas
        image = Image.open(self.images[idx]).convert("RGB")
        target_or = Image.open(self.segs[idx])

        # Muitos algoritmos esperam que o fundo trenha índice 0 e o objeto de interesse
        # possua valor 1. As imagens deste dataset possuem valores 1 para o objeto,
        # 2 para o fundo e 3 para píxeis indefinidos. Portanto, modificaremos os valores
        # 2->0 e 3->ignore_val
        target_np = np.array(target_or)
        target_np[target_np==2] = 0

        # Valor a ser usado para pixeis ignorados:
        if self.ignore_val!=3:
            target_np[target_np==3] = self.ignore_val

        # Padronizamos que um dataset retorna uma imagem pillow, então precisamos
        # converter novamente a imagem para esse formato
        target = Image.fromarray(target_np, mode="L")

        if self.transforms and apply_transform:
            image, target = self.transforms(image, target)

        return image, target
    
    def __len__(self):
        return len(self.images)

class TransformsTrain:

    def __init__(self, resize_size=384):
    
        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.RandomResizedCrop(size=(resize_size,resize_size), scale=(0.5,1.), 
                                     ratio=(0.9,1.1), antialias=True),
            #transf.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.01),
            transf.RandomHorizontalFlip(),
            transf.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img, target):
        # Convertemos os tensores para os tipos Image e Mask para que as transformações 
        # tratem adequadamente as duas imagens. Por exemplo, não faz sentido modificar
        # o brilho da imagem de rótulo
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        # Remoção da dimensão de canal do target, pois as funções de loss esperam
        # um target de tamanho batch_size x altura x largura
        target = target.squeeze()
        return img, target

class TransformsEval:

    def __init__(self, resize_size=384):

        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.Resize(size=resize_size, antialias=True),
            transf.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img, target):
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data
        target = target.squeeze()
        return img, target
    
def cat_list(images, fill_value=0):

    # Rótulos não possuem a dimensão de canal
    is_target = images[0].ndim==2

    num_rows, num_cols = zip(*[img.shape[-2:] for img in images])
    # Maior número de linhas e colunas no batch
    r_max, c_max = max(num_rows), max(num_cols)
    # Tamanho total do batch
    if is_target:
        batch_shape = (len(images), r_max, c_max)
    else:
        batch_shape = (len(images), 3, r_max, c_max)

    # Inserção de cada imagem dentro do batch
    batched_imgs = torch.full(batch_shape, fill_value, dtype=images[0].dtype)
    for idx in range(len(images)):
        img = images[idx]
        # Insere img na região de mesmo tamanho dentro do batch
        if is_target:
            batched_imgs[idx, :img.shape[0], :img.shape[1]] = img
        else:
            batched_imgs[idx, :, :img.shape[1], :img.shape[2]] = img

    return batched_imgs

def collate_fn(batch, img_fill=0, target_fill=2):

    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=img_fill)
    batched_targets = cat_list(targets, fill_value=target_fill)

    return batched_imgs, batched_targets

def unormalize(img):
    img = img.permute(1, 2, 0)
    mean = torch.tensor([122.7, 114.6, 100.9])
    std = torch.tensor([59.2, 58.4, 59.0])
    img = img*std + mean
    img = img.to(torch.uint8)

    return img

def get_dataset(root, split=0.2, resize_size=384):

    # Pesos das classes foram recalculados considerando o número de pixeis
    # de fundo e de objeto
    class_weights = (0.33, 0.67)

    ds = OxfordIIITPetSeg(root)
    n = len(ds)
    n_valid = int(n*split)

    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    
    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval(resize_size))

    return ds_train, ds_valid, class_weights