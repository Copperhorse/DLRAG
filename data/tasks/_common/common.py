"""Shared test fixtures. Copied next to every task's tests.py by the sandbox."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n=64, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, 1).astype(np.float32)
    y = (1 + 2 * x + 0.1 * rng.randn(n, 1)).astype(np.float32)
    return x, y


def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1, 1))


def make_loader(batch_size=16, n=64, seed=0, shuffle=False):
    x, y = make_data(n, seed)
    ds = TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def same_params(a, b):
    return all(torch.equal(p, q) for p, q in zip(a, b))


def bare(cls, **attrs):
    """Instance of ``cls`` WITHOUT running its __init__, with the given attributes set."""
    obj = cls.__new__(cls)
    obj.__dict__.update(attrs)
    return obj
