'''Script para o treinamento de uma rede de segmentação. As únicas
modificações deste script em relação ao de classificação são:

1. o uso do parâmetro ignore_index=2 na classe CrossEntropyLoss.
Ele faz com que os pixeis com valor 2 na imagem de rótulos sejam ignorados.

2. inclusão da função collate_fn no dataloader de validação, pois
as imagens não possuem o mesmo tamanho. 

3. Modificação da função de acurácia para medir a intersecção sobre a união.'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import get_dataset, collate_fn

# Gambiarra para importar o script train.py feito anteriormente
import sys
sys.path.insert(0, '../')
import M06_classificacao_de_imagens_naturais.train as train_class 

@torch.no_grad()
def iou(scores, targets):
    '''Função que calcula a Intersecção sobre a União entre o resultado
    da rede e o rótulo conhecido.'''

    # Transforma a predição da rede em índices 0 e 1, e aplica em reshape
    # nos tensores para transformá-los em 1D
    pred = scores.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)

    # Mantém apenas valores para os quais target!=2. O valor 2 indica píxeis
    # a serem ignorados
    pred = pred[targets!=2]
    targets = targets[targets!=2]

    # Verdadeiro positivos
    tp = ((targets==1) & (pred==1)).sum()
    # Verdadeiro negativos
    tn = ((targets==0) & (pred==0)).sum()
    # Falso positivos
    fp = ((targets==0) & (pred==1)).sum()
    # Falso negativos
    fn = ((targets==1) & (pred==0)).sum()

    # Algumas métricas interessantes para medir a qualidade do resultado
    # Fração de píxeis corretos
    acc = (tp+tn)/(tp+tn+fp+fn)
    # Intersecção sobre a união (IoU)
    iou = tp/(tp+fp+fn)
    # Precisão
    prec = tp/(tp+fp)
    # Revocação
    rev = tp/(tp+fn)

    # Retorna apenas o iou para não termos que reescrever a função de plotagem
    # dos resultados, que espera um único valor de performance
    return iou

def train(model, bs_train, bs_valid, num_epochs, lr, weight_decay=0., resize_size=224, seed=0,
          num_workers=5):

    train_class.seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_train, ds_valid, class_weights = get_dataset(
        '../data/oxford_pets', resize_size=resize_size
        )
    #ds_train.indices = ds_train.indices[:5*256]
    model.to(device)

    dl_train = DataLoader(ds_train, batch_size=bs_train, shuffle=True, 
                          num_workers=num_workers, persistent_workers=num_workers>0)
    # parâmetro collate_fn é necessário porque as imagens de validação possuem
    # tamanhos distintos
    dl_valid = DataLoader(ds_valid, batch_size=bs_valid, shuffle=False, 
                          collate_fn=collate_fn,
                          num_workers=num_workers, persistent_workers=num_workers>0)

    # ignore_index=2 indica que os pixeis com valor 2 na imagem de rótulos serão ignorados.
    # Isso inclui pixeis de borda e pixeis utilizados no padding da função collate_fn
    # acima
    loss_func = nn.CrossEntropyLoss(torch.tensor(class_weights, device=device), ignore_index=2)
    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9)
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    for epoch in range(0, num_epochs):
        loss_train = train_class.train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, perf = train_class.valid_step(model, dl_valid, loss_func, iou, device)
        logger.append((epoch, loss_train, loss_valid, perf))

        train_class.show_log(logger)

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
        torch.save(checkpoint, '../data/checkpoints/M07/checkpoint.pt')

        # Melhor modelo encontrado
        if loss_valid<best_loss:
            torch.save(checkpoint, '../data/checkpoints/M07/best_model.pt')
            best_loss = loss_valid

    model.to('cpu')

    return ds_train, ds_valid, logger