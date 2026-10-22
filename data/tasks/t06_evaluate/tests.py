import torch
from common import bare, make_loader
from solution import StepByStep


def test_behavior_returns_mean_loss_over_the_loader():
    vals = iter([1.0, 2.0, 3.0, 4.0])
    obj = bare(StepByStep, device="cpu", val_step_fn=lambda x, y: next(vals))
    assert abs(float(obj.evaluate(make_loader(batch_size=16, n=64))) - 2.5) < 1e-9


def test_convention_none_loader_returns_none():
    assert bare(StepByStep, device="cpu", val_step_fn=lambda x, y: 0.0).evaluate(None) is None


def test_convention_batches_are_moved_to_self_device():
    seen = []
    obj = bare(StepByStep, device="meta", val_step_fn=lambda x, y: seen.append((x.device.type, y.device.type)) or 0.0)
    obj.evaluate(make_loader())
    assert seen and all(s == ("meta", "meta") for s in seen)


def test_convention_runs_without_gradients():
    flags = []
    obj = bare(StepByStep, device="cpu", val_step_fn=lambda x, y: flags.append(torch.is_grad_enabled()) or 0.0)
    obj.evaluate(make_loader())
    assert flags and not any(flags)


def test_convention_plain_mean_of_batch_losses_not_sample_weighted():
    vals = iter([1.0, 2.0, 6.0])
    obj = bare(StepByStep, device="cpu", val_step_fn=lambda x, y: next(vals))
    assert abs(float(obj.evaluate(make_loader(batch_size=30, n=70))) - 3.0) < 1e-9
