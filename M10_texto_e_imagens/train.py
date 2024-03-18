"""
Esse script possui poucas mudanças em relação ao de treinamento para classificação.
As principais mudanças são:
1. Uso de uma collate_fn para criar batches de imagens e textos
2. Uso da clip_loss ao invés da entropia cruzada
3. Nas funções train_step e valid_step os textos não são enviados para a GPU,
pois isso é feito dentro do modelo TextEncoder
4. A função zero_shot_accuracy é utilizada para verificar a capacidade do modelo
em classificar as imagens
"""
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from IPython import display
from dataset import get_dataset, collate_fn
from model import clip_loss

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
    #ax1.set_ylim((0,1.))
    ax1.legend()
    ax2.plot(epochs, accs, '-o', ms=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim((0,1.))
    fig.tight_layout()

    display.clear_output(wait=True) 
    plt.show()

def train_step(model, dl_train, optim, loss_func, scheduler, device):
    '''Executa uma época de treinamento. Não podemos utilizar a função do
    script de treinamento de classificação apenas porque target.to() daria
    erro pois target não é um tensor.'''

    model.train()
    loss_log = 0.
    for imgs, texts in dl_train:
        imgs = imgs.to(device)
        optim.zero_grad()
        similarity = model(imgs, texts)
        loss = loss_func(similarity)
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
def zero_shot_accuracy(model, imgs, texts, label_embeds, device):

    # Estimativa da classe correta da imagem baseado no caption
    targets = []
    for text in texts:
        if 'cat' in text or 'kitten' in text:
            target = 0
        elif 'dog' in text or 'puppy' in text:
            target = 1
        else:
            target = 2
        targets.append(target)
    targets = torch.tensor(targets, device=device)

    # Projeção das imagens
    image_embeds = model.project_images(imgs)
    # Similaridade entre cada imagem e as palavras 'cat' e 'dog'
    scores = torch.matmul(image_embeds, label_embeds.t())
    # Índice da classe mais provável
    predictions = scores.argmax(dim=1)
    # Fração das imagens da classe gato (cachorro) que são mais similares à 
    # palavra 'cat' ('dog')
    mask = targets!=2
    targets = targets[mask]
    predictions = predictions[mask]
    acc = (predictions==targets).float().mean()

    return acc

# Anotador para evitar que gradientes sejam registrados dentro da função
@torch.no_grad()
def valid_step(model, dl_valid, loss_func, perf_func, device):

    # Coloca o modelo em modo de validação. 
    model.eval()
    label_embeds = model.project_texts(['cat', 'dog']).to(device)
    # Variáveis que armazenarão a loss e a acurácia
    loss_log = 0.
    perf_log = 0.
    for imgs, texts in dl_valid:
        imgs = imgs.to(device)
        similarity = model(imgs, texts)
        loss = loss_func(similarity)
        perf = perf_func(model, imgs, texts, label_embeds, device)

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

    ds_train, ds_valid = get_dataset('../data/oxford_pets', '../data/oxford_pets_captions.txt',
                                    resize_size=resize_size)
    model.to(device)

    # persistent_workers evita que cada processo reimporte as bibliotecas
    # Python no Windows
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                          num_workers=num_workers, persistent_workers=num_workers>0)
    dl_valid = DataLoader(ds_valid, batch_size=bs, shuffle=False, collate_fn=collate_fn,
                          num_workers=num_workers, persistent_workers=num_workers>0)

    loss_func = clip_loss
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9)
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        loss_train = train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, perf = valid_step(model, dl_valid, loss_func, zero_shot_accuracy, device)
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
        torch.save(checkpoint, '../data/checkpoints/M10/checkpoint.pt') 

        # Melhor modelo encontrado
        if loss_valid<best_loss:
            torch.save(checkpoint, '../data/checkpoints/M10/best_model.pt')
            best_loss = loss_valid             

    model.to('cpu')

    return ds_train, ds_valid, logger

