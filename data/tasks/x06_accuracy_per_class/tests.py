import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from stepbystep.v4 import StepByStep
from solution import accuracy_per_class


def _setup():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0]])); model.bias.zero_()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 2, generator=g)
    y = (torch.rand(40, generator=g) > 0.5).float().view(-1, 1)
    sbs = StepByStep(model, nn.BCEWithLogitsLoss(), optim.SGD(model.parameters(), lr=0.1))
    return sbs, DataLoader(TensorDataset(x, y), batch_size=16), x, y


def _expected(x, y):
    pred = (torch.sigmoid(x[:, 0:1]) > 0.5).float()
    return torch.stack([(pred[y == c] == c).float().mean() for c in (0.0, 1.0)])


def test_behavior_values_match_a_manual_computation():
    sbs, loader, x, y = _setup()
    out = accuracy_per_class(sbs, loader)
    assert torch.allclose(out.float(), _expected(x, y), atol=1e-6)


def test_convention_returns_a_tensor_with_one_entry_per_class():
    sbs, loader, _, _ = _setup()
    out = accuracy_per_class(sbs, loader)
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (2,)


def test_convention_model_left_in_training_mode():
    sbs, loader, _, _ = _setup()
    sbs.model.eval()
    accuracy_per_class(sbs, loader)
    assert sbs.model.training
