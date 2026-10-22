import torch, torch.nn as nn
from common import make_model, make_data, snapshot, same_params
from solution import make_val_step_fn


def _setup():
    model = make_model()
    x, y = make_data(16)
    return model, torch.as_tensor(x), torch.as_tensor(y), make_val_step_fn(model, nn.MSELoss())


def test_behavior_returns_loss_and_keeps_parameters():
    model, x, y, step = _setup()
    before = snapshot(model)
    loss = step(x, y)
    expected = nn.MSELoss()(model(x), y).item()
    assert abs(loss - expected) < 1e-6
    assert same_params(before, snapshot(model))


def test_convention_returns_python_float():
    _, x, y, step = _setup()
    assert type(step(x, y)) is float


def test_convention_sets_eval_mode():
    model, x, y, step = _setup()
    model.train()
    step(x, y)
    assert not model.training


def test_convention_no_gradients_computed():
    model, x, y, step = _setup()
    step(x, y)
    assert all(p.grad is None for p in model.parameters())
