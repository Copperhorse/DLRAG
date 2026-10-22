import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from helpers import index_splitter

def make_loaders(x, y, splits=(80, 20), batch_size=16):
    dataset = TensorDataset(torch.as_tensor(x).float(), torch.as_tensor(y).float())
    train_idx, val_idx = index_splitter(len(dataset), list(splits))
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
    return train_loader, val_loader
