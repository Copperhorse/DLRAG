import torch
from torch.utils.data import DataLoader, TensorDataset
from helpers import make_balanced_sampler

def balanced_train_loader(x, y, batch_size):
    x_t = torch.as_tensor(x).float()
    y_t = torch.as_tensor(y).float()
    sampler = make_balanced_sampler(y_t)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, sampler=sampler)
