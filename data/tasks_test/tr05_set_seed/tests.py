import random, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from common import bare
from solution import StepByStep


def _loader(generator=True):
    ds = TensorDataset(torch.arange(10).float().view(-1, 1))
    if generator:
        s = WeightedRandomSampler(torch.ones(10), 10, replacement=True, generator=torch.Generator())
        return DataLoader(ds, batch_size=5, sampler=s)
    return DataLoader(ds, batch_size=5)


def test_behavior_random_numbers_are_reproducible_for_all_three_libraries():
    obj = bare(StepByStep, train_loader=None)
    obj.set_seed(7); a = (torch.rand(3), np.random.rand(3), random.random())
    obj.set_seed(7); b = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    obj.set_seed(8)
    assert not torch.equal(torch.rand(3), a[0])


def test_convention_the_sampler_generator_of_the_train_loader_is_reseeded():
    obj = bare(StepByStep, train_loader=_loader())
    obj.set_seed(5)
    ref = torch.Generator().manual_seed(5)
    assert torch.equal(torch.rand(3, generator=obj.train_loader.sampler.generator), torch.rand(3, generator=ref))


def test_convention_never_fails_without_a_usable_loader_and_sets_cudnn_flags():
    for loader in (None, _loader(generator=False)):
        bare(StepByStep, train_loader=loader).set_seed(1)
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
