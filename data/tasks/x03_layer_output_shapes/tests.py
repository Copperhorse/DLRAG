import torch, torch.nn as nn, torch.optim as optim
from stepbystep.v4 import StepByStep
from common import make_loader
from solution import layer_output_shapes


def _sbs():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
    return StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.1))


def test_behavior_shapes_cover_all_batches():
    shapes = layer_output_shapes(_sbs(), make_loader(batch_size=16, n=60), ["0", "2"])
    assert shapes == {"0": (60, 4), "2": (60, 1)}


def test_convention_hooks_are_removed_and_model_back_in_train_mode():
    sbs = _sbs()
    layer_output_shapes(sbs, make_loader(), ["0", "2"])
    assert sbs.handles == {}
    assert not any(m._forward_hooks for m in sbs.model.modules())
    assert sbs.model.training


def test_convention_only_requested_layers_are_reported():
    shapes = layer_output_shapes(_sbs(), make_loader(), ["1"])
    assert list(shapes) == ["1"] and shapes["1"] == (64, 4)
