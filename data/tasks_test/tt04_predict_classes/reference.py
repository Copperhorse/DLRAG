import torch

class StepByStep:
    def predict_classes(self, x, threshold=.5):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        probs = torch.sigmoid(self.model(x_tensor.to(self.device)))
        self.model.train()
        return (probs > threshold).long().detach().cpu().numpy()
