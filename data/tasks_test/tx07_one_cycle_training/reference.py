import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from stepbystep.v4 import StepByStep

def train_one_cycle(model, train_loader, val_loader, n_epochs, max_lr):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    sbs = StepByStep(model, nn.MSELoss(), optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.set_lr_scheduler(OneCycleLR(optimizer, max_lr=max_lr, total_steps=n_epochs * len(train_loader)))
    sbs.train(n_epochs)
    return sbs
