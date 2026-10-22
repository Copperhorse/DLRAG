import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from stepbystep.v4 import StepByStep

def train_with_tricks(model, train_loader, val_loader, n_epochs, max_norm, step_size):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    sbs = StepByStep(model, nn.MSELoss(), optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.set_clip_grad_norm(max_norm)
    sbs.set_lr_scheduler(StepLR(optimizer, step_size=step_size, gamma=0.5))
    sbs.train(n_epochs)
    return sbs
