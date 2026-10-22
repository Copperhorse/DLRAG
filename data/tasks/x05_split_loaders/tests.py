import numpy as np, torch
from torch.utils.data import SubsetRandomSampler
from helpers import index_splitter
from solution import make_loaders


def _xy(n=100):
    r = np.random.RandomState(1)
    return r.rand(n, 1), r.rand(n, 1)


def _idx(loader):
    return sorted(int(i) for i in loader.sampler.indices)


def test_behavior_disjoint_parts_with_the_requested_sizes_over_one_dataset():
    tl, vl = make_loaders(*_xy())
    assert tl.dataset is vl.dataset and len(tl.dataset) == 100
    assert len(_idx(tl)) == 80 and len(_idx(vl)) == 20
    assert set(_idx(tl)).isdisjoint(_idx(vl))
    assert sorted(_idx(tl) + _idx(vl)) == list(range(100))


def test_convention_indices_are_the_books_seeded_split():
    tl, vl = make_loaders(*_xy(), splits=(70, 30))
    train_idx, val_idx = index_splitter(100, [70, 30])
    assert _idx(tl) == sorted(int(i) for i in train_idx) and _idx(vl) == sorted(int(i) for i in val_idx)


def test_convention_subset_random_samplers_and_float_tensors():
    tl, vl = make_loaders(*_xy())
    assert isinstance(tl.sampler, SubsetRandomSampler) and isinstance(vl.sampler, SubsetRandomSampler)
    xb, yb = next(iter(tl))
    assert xb.dtype == torch.float32 and yb.dtype == torch.float32 and tl.batch_size == 16
