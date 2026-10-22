import torch, torch.nn as nn, torch.optim as optim
from common import make_model, make_data, snapshot, same_params
from solution import make_accum_train_step_fn


def _setup(k=2):
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=0.1)
    x, y = make_data(16)
    return model, opt, torch.as_tensor(x), torch.as_tensor(y), make_accum_train_step_fn(model, nn.MSELoss(), opt, k)


def test_behavior_updates_only_every_k_calls_with_averaged_gradient():
    model, opt, x, y, step = _setup(2)
    start = snapshot(model)
    step(x, y)
    assert same_params(start, snapshot(model))
    step(x, y)
    ref = make_model(); ropt = optim.SGD(ref.parameters(), lr=0.1)
    nn.MSELoss()(ref(x), y).backward(); ropt.step()
    assert not same_params(start, snapshot(model))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(snapshot(model), snapshot(ref)))


def test_convention_returns_unscaled_python_float_loss():
    model, _, x, y, step = _setup(4)
    loss = step(x, y)
    assert type(loss) is float and abs(loss - nn.MSELoss()(model(x), y).item()) < 1e-6


def test_convention_gradients_zeroed_right_after_the_update():
    model, _, x, y, step = _setup(2)
    step(x, y); step(x, y)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_convention_sets_train_mode():
    model, _, x, y, step = _setup(2)
    model.eval(); step(x, y)
    assert model.training
