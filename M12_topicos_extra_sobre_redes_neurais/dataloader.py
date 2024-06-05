import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    '''Dataset básico para usar no dataloader'''

    def __init__(self, vals):
        self.vals = vals

    def __getitem__(self, idx):
        '''Esta função retorna dados sobre os processos'''
        info = torch.utils.data.get_worker_info()

        return f'{idx=} processado por worker {info.id}'
    
    def __len__(self):
        return len(self.vals)
    
def collate_fn(batch):
    '''collate para não dar erro ao criar o batch'''
    return batch
    
def main():

    ds = MyDataset(list(range(15)))
    dl = DataLoader(ds, batch_size=4, num_workers=3, collate_fn=collate_fn)

    for batch_idx, batch in enumerate(dl):
        print(f'{batch_idx=}')
        print(batch)
        