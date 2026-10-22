import torch.nn as nn

class Decoder(nn.Module):
    def init_hidden_mean(self, hidden_seq):
        hidden_mean = hidden_seq.mean(dim=1, keepdim=True)   # N, 1, H
        self.hidden = hidden_mean.permute(1, 0, 2)           # 1, N, H
