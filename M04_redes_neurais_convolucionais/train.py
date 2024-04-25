import matplotlib.pyplot as plt
from IPython import display
import torch
from torch import nn
from torch.utils.data import DataLoader

def train_step(model, dl_train, optim, loss_func):
    '''Executa uma época de treinamento.'''

    # Coloca o modelo em modo treinamento. Neste notebook não fará diferença,
    # mas algumas camadas (batchnorm, dropout) precisam disso
    model.train()
    # Armazenará a média das losses de todos os bathces
    loss_log = 0.
    for imgs, targets in dl_train:
        optim.zero_grad()
        scores = model(imgs)
        loss = loss_func(scores, targets)
        loss.backward()
        optim.step()

        # Multiplica por imgs.shape[0] porque o último batch pode ter tamanho diferente
        loss_log += loss.detach()*imgs.shape[0]

    # Média das losses calculadas
    loss_log /= len(dl_train.dataset)

    return loss_log.item()

def accuracy(scores, targets):
    return (scores.argmax(dim=1)==targets).float().mean()

# Anotador para evitar que gradientes sejam registrados dentro da função
@torch.no_grad()
def valid_step(model, dl_valid, loss_func, perf_func):
    '''Valida o modelo no conjunto de validação.'''

    # Coloca o modelo em modo de validação.
    model.eval()
    # Variáveis que armazenarão a loss e a acurácia
    loss_log = 0.
    perf_log = 0.
    for imgs, targets in dl_valid:
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

def show_log(logger):
    '''Mostra as métricas de treinamento e validação.'''

    epochs, losses_train, losses_valid, accs = zip(*logger)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,3))
    ax1.plot(epochs, losses_train, label='Train loss')
    ax1.plot(epochs, losses_valid, label='Valid loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(epochs, accs)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    fig.tight_layout()

    display.clear_output(wait=True)
    plt.show()

def train(model, ds_train, ds_valid, bs, num_epochs, lr, perf_func=accuracy,
          weight_decay=0.):

    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=bs, shuffle=False)

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger = []
    for epoch in range(0, num_epochs):
        loss_train = train_step(model, dl_train, optim, loss_func)
        loss_valid, perf = valid_step(model, dl_valid, loss_func, perf_func)
        logger.append((epoch, loss_train, loss_valid, perf))
        show_log(logger)

    return logger
