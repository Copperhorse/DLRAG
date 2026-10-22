import torch.nn as nn
import torch.optim as optim
from stepbystep.v4 import StepByStep

def resume_training(checkpoint_path, model, train_loader, val_loader, extra_epochs):
    sbs = StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.1))
    sbs.set_loaders(train_loader, val_loader)
    sbs.load_checkpoint(checkpoint_path)
    sbs.train(extra_epochs)
    return sbs
