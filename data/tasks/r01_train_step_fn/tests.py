import torch, torch.nn as nn, torch.optim as optim
from common import make_model, make_data, snapshot, same_params
from solution import make_train_step_fn


def _setup():
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=0.1)
    x, y = make_data(16)
    return model, opt, torch.as_tensor(x), torch.as_tensor(y), make_train_step_fn(model, nn.MSELoss(), opt)


def test_behavior_updates_parameters_and_lowers_loss():
    model, opt, x, y, step = _setup()
    before = snapshot(model)
    first = step(x, y)
    assert not same_params(before, snapshot(model))
    for _ in range(50):
        last = step(x, y)
    assert last < first


def test_convention_returns_python_float():
    _, _, x, y, step = _setup()
    assert type(step(x, y)) is float


def test_convention_gradients_zeroed_after_step():
    model, _, x, y, step = _setup()
    step(x, y)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_convention_sets_train_mode():
    model, _, x, y, step = _setup()
    model.eval()
    step(x, y)
    assert model.training
