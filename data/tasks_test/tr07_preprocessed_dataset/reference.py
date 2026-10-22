import torch
from torch.utils.data import TensorDataset

def preprocessed_dataset(model, loader, device=None):
    if device is None:
        device = next(model.parameters()).device
    features = None
    labels = None
    for i, (x, y) in enumerate(loader):
        model.eval()
        x = x.to(device)
        output = model(x)
        if i == 0:
            features = output.detach().cpu()
            labels = y.cpu()
        else:
            features = torch.cat([features, output.detach().cpu()])
            labels = torch.cat([labels, y.cpu()])
    dataset = TensorDataset(features, labels)
    return dataset
