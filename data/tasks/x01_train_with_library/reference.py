import torch.nn as nn
import torch.optim as optim
from stepbystep.v4 import StepByStep

def train_regression(model, train_loader, val_loader, n_epochs):
    sbs = StepByStep(model, nn.MSELoss(reduction='mean'), optim.SGD(model.parameters(), lr=0.1))
    sbs.set_loaders(train_loader, val_loader)
    sbs.train(n_epochs)
    return sbs
