import torch
from torch.utils.data import DataLoader, TensorDataset
from data_generation.ball import load_data

def ball_loader(n_points, n_dims):
    X, y = load_data(n_dims=n_dims, n_points=n_points)
    dataset = TensorDataset(torch.as_tensor(X).float(), torch.as_tensor(y).float())
    return DataLoader(dataset, batch_size=len(X))
