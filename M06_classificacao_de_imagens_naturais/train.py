import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from IPython import display
from dataset import get_dataset

def seed_all(seed):
    "Semente para o pytorch, numpy e python."
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def show_log(logger):
    """Plota métricas em um notebook."""

    epochs, losses_train, losses_valid, accs = zip(*logger)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,3))
    ax1.plot(epochs, losses_train, '-o', ms=2, label='Train loss')
    ax1.plot(epochs, losses_valid, '-o', ms=2, label='Valid loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim((0,1.))
    ax1.legend()
    ax2.plot(epochs, accs, '-o', ms=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim((0,1.))
    fig.tight_layout()

    display.clear_output(wait=True)
    plt.show()

def train_step(model, dl_train, optim, loss_func, scheduler, device):
    '''Executa uma época de treinamento.'''

    # Coloca o modelo em modo treinamento. Neste notebook não fará diferença,
    # mas algumas camadas (batchnorm, dropout) precisam disso
    model.train()
    # Armazenará a média das losses de todos os bathces
    loss_log = 0.
    for imgs, targets in dl_train:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optim.zero_grad()
        scores = model(imgs)
        loss = loss_func(scores, targets)
        loss.backward()
        optim.step()

        # Multiplica por imgs.shape[0] porque o último batch pode ter tamanho diferente
        loss_log += loss.detach()*imgs.shape[0]

    # Muda o learning rate
    scheduler.step()

    # Média das losses calculadas
    loss_log /= len(dl_train.dataset)

    return loss_log.item()

@torch.no_grad()
def accuracy(scores, targets):
    return (scores.argmax(dim=1)==targets).float().mean()

# Anotador para evitar que gradientes sejam registrados dentro da função
@torch.no_grad()
def valid_step(model, dl_valid, loss_func, perf_func, device):

    # Coloca o modelo em modo de validação.
    model.eval()
    # Variáveis que armazenarão a loss e a acurácia
    loss_log = 0.
    perf_log = 0.
    for imgs, targets in dl_valid:
        imgs = imgs.to(device)
        targets = targets.to(device)
        scores = model(imgs)
        loss = loss_func(scores, targets)
        perf = perf_func(scores, targets)

        # Multiplica por imgs.shape[0] porque o último batch pode ter tamanho diferente
        loss_log += loss*imgs.shape[0]
        perf_log += perf*imgs.shape[0]

    # Média dos valores calculados
    loss_log /= len(dl_valid.dataset)
    perf_log /= len(dl_valid.dataset)

    return loss_log.item(), perf_log.item()

def train(model, bs, num_epochs, lr, weight_decay=0., resize_size=224, seed=0,
          num_workers=5):

    # Fixa todas as seeds
    seed_all(seed)
    # Usa cuda se disponível
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Carrega o dataset
    ds_train, ds_valid, class_weights = get_dataset('../data/oxford_pets', resize_size=resize_size)
    # Truque para testar o código, fingimos que o dataset possui menos imagens
    #ds_train.indices = ds_train.indices[:5*256]
    model.to(device)

    # persistent_workers evita que cada processo reimporte as bibliotecas
    # Python no Windows
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True,
                          num_workers=num_workers, persistent_workers=num_workers>0)
    dl_valid = DataLoader(ds_valid, batch_size=bs, shuffle=False,
                          num_workers=num_workers, persistent_workers=num_workers>0)

    loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=device))
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9)  # momentum suaviza o gradiente
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        loss_train = train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, perf = valid_step(model, dl_valid, loss_func, accuracy, device)
        logger.append((epoch, loss_train, loss_valid, perf))

        show_log(logger)

        # Dados sobre o estado atual
        checkpoint = {
            'params':{'bs':bs,'lr':lr,'weight_decay':weight_decay},
            'model':model.state_dict(),
            'optim':optim.state_dict(),
            'sched':sched.state_dict(),
            'logger':logger
        }

        # Salva o estado atual
        torch.save(checkpoint, '../data/checkpoints/M06/checkpoint.pt')

        # Melhor modelo encontrado
        if loss_valid<best_loss:
            torch.save(checkpoint, '../data/checkpoints/M06/best_model.pt')
            best_loss = loss_valid

    model.to('cpu')

    return ds_train, ds_valid, logger
