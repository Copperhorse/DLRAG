import torch.nn as nn
import torch.optim as optim
from stepbystep.v4 import StepByStep

def train_with_tensorboard(model, train_loader, val_loader, n_epochs, run_name, folder):
    sbs = StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.1))
    sbs.set_loaders(train_loader, val_loader)
    sbs.set_tensorboard(run_name, folder)
    sbs.train(n_epochs)
    return sbs
