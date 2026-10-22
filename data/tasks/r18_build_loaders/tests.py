import numpy as np, torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from solution import build_loaders


def _xy(n=100):
    r = np.random.RandomState(1)
    return r.rand(n, 1), r.rand(n, 1)


def test_behavior_sizes_batches_and_samplers():
    tl, vl = build_loaders(*_xy())
    assert isinstance(tl, DataLoader) and isinstance(vl, DataLoader)
    assert len(tl.dataset) == 80 and len(vl.dataset) == 20
    assert tl.batch_size == 16 and vl.batch_size == 16
    assert isinstance(tl.sampler, RandomSampler) and isinstance(vl.sampler, SequentialSampler)


def test_convention_float_tensors():
    tl, _ = build_loaders(*_xy())
    xb, yb = next(iter(tl))
    assert xb.dtype == torch.float32 and yb.dtype == torch.float32


def test_convention_split_is_the_books_seeded_random_split():
    x, y = _xy()
    tl, vl = build_loaders(x, y)
    torch.manual_seed(13)
    ds = TensorDataset(torch.as_tensor(x).float(), torch.as_tensor(y).float())
    ref_train, ref_val = random_split(ds, [80, 20])
    assert list(tl.dataset.indices) == list(ref_train.indices)
    assert list(vl.dataset.indices) == list(ref_val.indices)
