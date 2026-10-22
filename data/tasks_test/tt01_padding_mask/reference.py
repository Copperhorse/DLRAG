import torch

def padding_mask(lengths, max_len):
    lengths = torch.as_tensor(lengths)
    positions = torch.arange(max_len).view(1, 1, max_len)
    return positions < lengths.view(-1, 1, 1)
