import torch

class StepByStep:
    def train(self, n_epochs, seed=42, patience=None):
        self.set_seed(seed)
        best = float('inf')
        bad_epochs = 0
        for epoch in range(n_epochs):
            self.total_epochs += 1
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            if patience is not None:
                if val_loss < best:
                    best = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break
