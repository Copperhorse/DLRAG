import torch
from common import make_loader
from solution import StepByStep


def _func(x, y):
    return torch.tensor([len(x), 1])


def test_behavior_sum_over_batches_callable_on_class():
    out = StepByStep.loader_apply(make_loader(batch_size=16, n=64), _func)
    assert out.tolist() == [64, 4]


def test_convention_default_reduce_is_sum():
    assert StepByStep.loader_apply(make_loader(), _func).tolist() == StepByStep.loader_apply(make_loader(), _func, reduce="sum").tolist()


def test_convention_mean_is_float_and_per_element():
    out = StepByStep.loader_apply(make_loader(batch_size=16, n=64), _func, reduce="mean")
    assert out.dtype.is_floating_point and out.tolist() == [16.0, 1.0]


def test_convention_other_reduce_returns_stacked_per_batch_results():
    out = StepByStep.loader_apply(make_loader(batch_size=16, n=64), _func, reduce=None)
    assert tuple(out.shape) == (4, 2)
