import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_generation.square_sequences import generate_sequences

def square_sequence_loader(n, batch_size=16):
    points, directions = generate_sequences(n=n)
    X = torch.as_tensor(np.array(points)).float()
    y = torch.as_tensor(directions).view(-1, 1).float()
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)
