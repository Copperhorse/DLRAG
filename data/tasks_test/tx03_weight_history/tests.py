import torch.nn as nn, torch.optim as optim
from stepbystep.v4 import StepByStep
from common import make_loader
from solution import weight_history


def _sbs():
    import torch
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1))
    sbs = StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.05))
    sbs.set_loaders(make_loader(batch_size=16, n=64, shuffle=True), make_loader(batch_size=16, n=64, seed=1))
    return sbs


def test_behavior_one_recorded_value_per_forward_pass_for_each_parameter():
    hist = weight_history(_sbs(), "0", 2)          # 2 epochs x (4 training + 4 validation) forward passes
    assert set(hist) == {"weight", "bias"}
    assert len(hist["weight"]) == 16 and len(hist["bias"]) == 16
    assert len(hist["weight"][0]) == 3 and len(hist["weight"][0][0]) == 1 and len(hist["bias"][0]) == 3


def test_convention_recorded_weights_change_during_training():
    hist = weight_history(_sbs(), "2", 2)
    assert hist["weight"][0] != hist["weight"][-1]


def test_convention_no_hook_is_left_attached_afterwards():
    sbs = _sbs()
    weight_history(sbs, "0", 1)
    assert sbs.handles == {} and not any(m._forward_hooks for m in sbs.model.modules())
