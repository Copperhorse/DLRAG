import torch, torch.nn as nn, torch.optim as optim
from common import make_model, make_data, snapshot
from solution import make_clipped_train_step_fn


def _setup(max_norm, lr=1.0):
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=lr)
    x, y = make_data(16)
    x, y = torch.as_tensor(x) * 50, torch.as_tensor(y) * 50     # large gradients
    return model, torch.as_tensor(x), torch.as_tensor(y), make_clipped_train_step_fn(model, nn.MSELoss(), opt, max_norm)


def test_behavior_update_size_is_bounded_by_lr_times_max_norm():
    model, x, y, step = _setup(0.5, lr=1.0)
    before = snapshot(model)
    step(x, y)
    delta = torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(before, snapshot(model))))
    assert 0.49 < float(delta) < 0.51


def test_convention_returns_python_float_of_unclipped_loss():
    model, x, y, step = _setup(0.5)
    expected = nn.MSELoss()(model(x), y).item()
    out = step(x, y)
    assert type(out) is float and abs(out - expected) < 1e-4 * max(1.0, expected)


def test_convention_gradients_zeroed_after_step_and_train_mode():
    model, x, y, step = _setup(0.5)
    model.eval(); step(x, y)
    assert model.training
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())
