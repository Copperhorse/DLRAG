import torch
from torch.utils.data import random_split

def index_splitter(n, splits, seed=13):
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    total = splits_tensor.sum().float()
    if not total.isclose(torch.ones(1)[0]):
        splits_tensor = splits_tensor / total
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)
