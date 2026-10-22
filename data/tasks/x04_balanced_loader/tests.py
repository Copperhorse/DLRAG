import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from solution import balanced_train_loader


def _xy():
    rng = np.random.RandomState(0)
    y = np.concatenate([np.zeros(180), np.ones(20)]).reshape(-1, 1)
    return rng.rand(200, 1), y


def test_behavior_classes_are_drawn_about_equally():
    x, y = _xy()
    loader = balanced_train_loader(x, y, 20)
    if isinstance(getattr(loader.sampler, "generator", None), torch.Generator):
        loader.sampler.generator.manual_seed(0)
    ys = torch.cat([yb for _, yb in loader])
    assert isinstance(loader, DataLoader) and len(ys) == 200
    assert 0.35 < ys.mean().item() < 0.65


def test_convention_float_tensors_and_batch_size():
    loader = balanced_train_loader(*_xy(), 20)
    xb, yb = next(iter(loader))
    assert xb.dtype == torch.float32 and yb.dtype == torch.float32 and loader.batch_size == 20


def test_convention_weighted_random_sampler_with_replacement_and_seedable_generator():
    s = balanced_train_loader(*_xy(), 20).sampler
    assert isinstance(s, WeightedRandomSampler) and s.replacement is True
    assert isinstance(s.generator, torch.Generator)
