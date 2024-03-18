"""script de treinamento de um autoencoder. As diferenças para o scritp de treinamento 
de classificação são:

1. A função de acurácia é modificada, pois não temos um problema de classificação. 
É um problema em aberto definir a similaridade entre imagens.

2. Utilizamos a classe L1Loss para calcular a loss.

3. Adicionamos ao logger as imagens que estão sendo reconstruídas pela rede, para
visualizar se o processo está evoluindo.  

A medida Fréchet Inception Distance poderia ser utilizada no treinamento da rede,
mas ela não foi utilizada neste script para simplificar o treinamento.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from IPython import display
from dataset import get_dataset

# Gambiarra para importar o script train.py feito anteriormente
import sys
sys.path.insert(0, '../')
import M06_classificacao_de_imagens_naturais.train as train_class 

# A acurácia é a própria loss L1 de validação que já está sendo calculada
# na função de treinamento. Definimos uma função que retorna 0 que será ignorado.
@torch.no_grad()
def accuracy(scores, targets):
    return torch.tensor(0.)

def show_log(logger):
    """Plota as losses e também exemplos de imagens de resultado."""

    epochs, losses_train, losses_valid, accs, imgs = zip(*logger)

    epochs = list(epochs)
    imgs = list(imgs)

    # Cria uma lista de 5 índices igualmente espaçados entre 0 e len(imgs)
    inds = torch.linspace(0, len(imgs)-1, 5)
    inds = inds.int()
    # Corta lista de índices caso len(img)<5
    inds = inds[:len(imgs)]

    fig, axs = plt.subplots(2, 3, figsize=(12,4))
    axs = axs.reshape(-1)
    axs[0].plot(epochs, losses_train, '-o', label='Train loss')
    axs[0].plot(epochs, losses_valid, '-o', label='Valid loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Mostra as imagens
    for i, ind in enumerate(inds):
        axs[i+1].imshow(imgs[ind].permute(1,2,0))
        axs[i+1].set_title(f"epoch={epochs[ind]}")

    fig.tight_layout()

    display.clear_output(wait=True) 
    plt.show()

def train(model, bs_train, bs_valid, num_epochs, lr, weight_decay=0., resize_size=224, seed=0, 
          num_workers=5):
    
    train_class.seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_train, ds_valid = get_dataset(
        '../data/oxford_pets', resize_size=resize_size
        )

    model.to(device)

    dl_train = DataLoader(ds_train, batch_size=bs_train, shuffle=True, 
                          num_workers=num_workers, persistent_workers=num_workers>0)
    dl_valid = DataLoader(ds_valid, batch_size=bs_valid, shuffle=False,
                          num_workers=num_workers, persistent_workers=num_workers>0)

    # Loss L1 ao invés da entropia cruzada.
    loss_func = nn.L1Loss()
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9) 
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        loss_train = train_class.train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, perf = train_class.valid_step(model, dl_valid, loss_func, accuracy, device)

        # Aplica o modelo em uma das imagens e adiciona o resultado ao logger. 
        # Fazer isso em todas as épocas cria um logger que precisa de bastante
        # espaço de armazenamento, o ideal seria fazer a cada x épocas.
        img, _ = ds_valid[1]
        img = img.to(device)
        with torch.no_grad():
            img_rec = model(img.unsqueeze(0))[0]
        img_rec = img_rec.to('cpu')
        logger.append((epoch, loss_train, loss_valid, perf, img_rec))

        show_log(logger)

        # Dados sobre o estado atual
        checkpoint = {
            'params':{'bs_train':bs_train,'bs_valid':bs_valid,'lr':lr,
                      'weight_decay':weight_decay},
            'model':model.state_dict(),
            'optim':optim.state_dict(),
            'sched':sched.state_dict(),
            'logger':logger
        }

        # Salva o estado atual
        torch.save(checkpoint, '../data/checkpoints/M08/checkpoint.pt') 

        # Melhor modelo encontrado
        if loss_valid<best_loss:
            torch.save(checkpoint, '../data/checkpoints/M08/best_model.pt')
            best_loss = loss_valid         

    model.to('cpu')

    return ds_train, ds_valid, logger