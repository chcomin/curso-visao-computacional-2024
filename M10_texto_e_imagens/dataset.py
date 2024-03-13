"""
1. `scale` modificado para 0.8 e `resize` para 224, para adicionar contexto
"""
import random
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as transf

# Gambiarra para importar as funções de dataset feitas anteriormente
import sys
sys.path.insert(0, '../')
from M06_classificacao_de_imagens_naturais.dataset import OxfordIIITPet, Subset, unormalize

class OxfordIIITPetCap(OxfordIIITPet):
    """Modifiação da classe OxfordIIITPet para retornar imagens e captions."""

    def __init__(self, root, cap_file, transforms=None):
        super().__init__(root)

        # Cria dicionário no qual a chave é o nome da imagem e o valor
        # é o caption
        text = open(cap_file).read().splitlines()
        cap_dict = {}
        for line in text:
            file, caption = line.split(',', 1)
            cap_dict[file.strip()] = caption.strip()
        
        # Para cada imagem, armazena o respectivo caption
        captions = []
        for filename in self.images:
            name = filename.stem
            captions.append(cap_dict[name])

        self.captions = captions
        self.transforms = transforms

    def __getitem__(self, idx):
        # Chama o método .__getitem__ da classe pai e ignora o target
        image, _ = super().__getitem__(idx)
        caption = self.captions[idx]

        if self.transforms is not None:
            image = self.transforms(image)

        # Retorna a imagem e o caption
        return image, caption
    
class TransformsTrain:

    def __init__(self, resize_size=224):

        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.RandomResizedCrop(size=(resize_size,resize_size), scale=(0.8,1.), 
                                     ratio=(0.7,1.3), antialias=True),
            transf.RandomHorizontalFlip(),
            transf.ToDtype(torch.float32),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(img)

class TransformsEval:

    def __init__(self):

        transforms = transf.Compose([
            transf.PILToTensor(),   
            transf.Resize(size=224, antialias=True),
            transf.CenterCrop(size=224),
            transf.ToDtype(torch.float32),
            transf.Normalize(mean=(122.7, 114.6, 100.9), std=(59.2, 58.4, 59.0))
        ])

        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(img)

def wrap_text(text):
    """Função para quebrar o texto em linhas. Usada apenas para visualização
    dos dados."""
    
    text_split = text.split()
    for idx in range(len(text_split)):
        if (idx+1)%4==0:
            text_split[idx] += '\n'
        else:
            text_split[idx] += ' '
    wrapped_text = ''.join(text_split)

    return wrapped_text

def show_items(ds):

    inds = torch.randint(0, len(ds), size=(12,))
    items = [ds[idx] for idx in inds]

    fig, axs = plt.subplots(2, 6, figsize=(12,5))
    axs = axs.reshape(-1)
    for idx in range(12):
        image, caption = items[idx]
        caption = wrap_text(caption)
        axs[idx].imshow(image.permute(1, 2, 0)/255.)
        axs[idx].set_title(caption, loc='center', wrap=True)
    fig.tight_layout()

def collate_fn(batch):
    """Concatena imagens, mas não os textos"""

    images, texts = list(zip(*batch))
    batched_imgs = torch.stack(images, 0)

    return batched_imgs, texts

def get_dataset(root, cap_file, split=0.2, resize_size=224):

    ds = OxfordIIITPetCap(root, cap_file)
    n = len(ds)
    n_valid = int(n*split)

    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    ds_train = Subset(ds, indices[n_valid:], TransformsTrain(resize_size))
    ds_valid = Subset(ds, indices[:n_valid], TransformsEval())

    return ds_train, ds_valid
