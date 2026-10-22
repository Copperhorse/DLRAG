import torch, torch.nn as nn, torch.optim as optim
from stepbystep.v4 import StepByStep
from common import make_loader
from solution import gradient_history


def _sbs():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1))
    sbs = StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.05))
    sbs.set_loaders(make_loader(batch_size=16, n=64, shuffle=True), make_loader(batch_size=16, n=64, seed=1))
    return sbs


def test_behavior_one_gradient_per_training_mini_batch_for_each_parameter():
    hist = gradient_history(_sbs(), "0", 2)         # 2 epochs x 4 training mini-batches
    assert set(hist) == {"weight", "bias"}
    assert len(hist["weight"]) == 8 and len(hist["bias"]) == 8
    assert len(hist["weight"][0]) == 3 and len(hist["bias"][0]) == 3


def test_convention_recording_is_switched_off_when_the_function_returns():
    sbs = _sbs()
    hist = gradient_history(sbs, "2", 1)
    n = len(hist["weight"])
    sbs.train(1)
    assert len(hist["weight"]) == n and sbs.handles == {}
