import torch
import torch.nn as nn

class StepByStep:
    def correct(self, x, y, threshold=.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()
        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)
