import random
import torch
import torchvision.transforms.v2 as transf
from torchvision import tv_tensors

# Gambiarra para importar o script dataset.py feito anteriormente
import sys
sys.path.insert(0, '../')
from M06_classificacao_de_imagens_naturais.dataset import OxfordIIITPet
# A classe Subset de classificação não aplica transformação no target, então
# vamos usar a classe Subset de segmentação
from M07_segmentacao.dataset import Subset

class OxfordIIITPetAe(OxfordIIITPet):

    def __init__(self, root):
        super().__init__(root)

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)

        return image, image

class AddNoise:

    def __init__(self, p_range):
        """
        Args:
            p_range (tuple): valor mínimo e máximo da probabilidade de remover um pixel.
        """

        self.p_range = p_range

    def __call__(self, img):

        p_range = self.p_range
        # Sorteia probabilidade de remoção global no intervalo (p_range[0], p_range[1])
        prob = (p_range[1]-p_range[0])*torch.rand(()) + p_range[0]
        # Gera matriz 2D 
        prob_mat = torch.full(img.shape[-2:], prob)
        # Para cada pixel, sorteia o valor 1 com probabilidade prob
        # e o valor 0 com probabilidade 1-prob
        noise = torch.bernoulli(prob_mat)
        out = img*noise

        return out

class TransformsTrain:

    def __init__(self, p_range, resize_size=384):
    
        add_noise = AddNoise(p_range)
        transforms = transf.Compose([
            transf.RandomResizedCrop(size=(resize_size,resize_size), scale=(0.5,1.), 
                                     ratio=(0.9,1.1), antialias=True),
            transf.RandomHorizontalFlip(),
            # Efetivamente, a transformação abaixo divide os valores por 255. Foi
            # usado 255.01 para evitar problemas de arredondamento
            transf.Normalize(mean=(0.,0.,0.), std=(255.01,255.01,255.01)), 
        ])  

        self.add_noise = add_noise
        self.transforms = transforms

    def __call__(self, img, target):

        # Precisamos aplicar todas as transformações em ambas as imagens, exceto
        # a transformação de ruído. Por isso primeiro convertemos as imagens para,
        # tensor float32, aplicamos ruído em `img` e depois as demais transformações
        # em ambas as imagens 
        img = transf.functional.pil_to_tensor(img)
        target = transf.functional.pil_to_tensor(target)
        img = img.to(torch.float32)
        target = target.to(torch.float32)
        img = self.add_noise(img)

        img = tv_tensors.Image(img)
        target = tv_tensors.Image(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data

        return img, target

class TransformsEval:

    def __init__(self, p=0.9, resize_size=384):

        transforms = transf.Compose([
            transf.Resize(size=(resize_size, resize_size), antialias=True),
            transf.Normalize(mean=(0.,0.,0.), std=(255.01,255.01,255.01))
        ])

        # Note o mesmo valor de p para mínimo e máximo ruído
        self.add_noise = AddNoise((p,p))
        self.transforms = transforms

    def __call__(self, img, target):

        img = transf.functional.pil_to_tensor(img)
        target = transf.functional.pil_to_tensor(target)
        img = img.to(torch.float32)
        target = target.to(torch.float32)
        img = self.add_noise(img)

        img = tv_tensors.Image(img)
        target = tv_tensors.Image(target)
        img, target = self.transforms(img, target)
        img = img.data
        target = target.data

        return img, target

def get_dataset(root, split=0.2, p_range=(0.7,0.9), resize_size=384):

    ds = OxfordIIITPetAe(root)
    n = len(ds)
    n_valid = int(n*split)

    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    
    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(p_range, resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval())

    return ds_train, ds_valid