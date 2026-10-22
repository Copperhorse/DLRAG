import numpy as np
import torch

class StepByStep:
    def evaluate(self, loader):
        if loader is None:
            return None
        losses = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                losses.append(self.val_step_fn(x_batch, y_batch))
        return np.mean(losses)
