import torch
from common import bare, make_model
from solution import StepByStep


def _obj_with_grads(scale=10.0):
    model = make_model()
    for p in model.parameters():
        p.grad = torch.full_like(p, scale)
    return bare(StepByStep, model=model, clipping=None)


def test_behavior_value_clipping_limits_each_gradient():
    obj = _obj_with_grads()
    obj.set_clip_grad_value(0.5)
    obj.clipping()
    assert all(float(p.grad.abs().max()) <= 0.5 + 1e-6 for p in obj.model.parameters())


def test_behavior_norm_clipping_limits_total_norm():
    obj = _obj_with_grads()
    obj.set_clip_grad_norm(1.0)
    obj.clipping()
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in obj.model.parameters()))
    assert float(total) <= 1.0 + 1e-5


def test_convention_configuring_does_not_clip_immediately():
    obj = _obj_with_grads()
    obj.set_clip_grad_value(0.5)
    assert all(float(p.grad.abs().max()) == 10.0 for p in obj.model.parameters())
    assert callable(obj.clipping)


def test_convention_remove_clip_resets_attribute_to_none():
    obj = _obj_with_grads()
    obj.set_clip_grad_norm(1.0)
    obj.remove_clip()
    assert obj.clipping is None


def test_convention_norm_type_defaults_to_two_and_is_configurable():
    a = _obj_with_grads(); a.set_clip_grad_norm(1.0); a.clipping()
    b = _obj_with_grads(); b.set_clip_grad_norm(1.0, norm_type=float("inf")); b.clipping()
    # grads are all 10: L2 norm = sqrt(200) -> 0.7071 each; inf norm = 10 -> 1.0 each
    assert all(abs(float(p.grad.abs().max()) - 0.70710678) < 1e-4 for p in a.model.parameters())
    assert all(abs(float(p.grad.abs().max()) - 1.0) < 1e-4 for p in b.model.parameters())
