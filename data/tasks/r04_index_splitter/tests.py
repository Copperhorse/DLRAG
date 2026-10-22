import torch
from torch.utils.data import random_split
from solution import index_splitter


def _indices(subset):
    return sorted(int(i) for i in (subset.indices if hasattr(subset, "indices") else subset))


def test_behavior_disjoint_cover_and_sizes():
    parts = index_splitter(100, [80, 20])
    a, b = _indices(parts[0]), _indices(parts[1])
    assert len(a) == 80 and len(b) == 20
    assert sorted(a + b) == list(range(100))


def test_behavior_reproducible_for_same_seed():
    assert _indices(index_splitter(50, [.7, .3], seed=5)[0]) == _indices(index_splitter(50, [.7, .3], seed=5)[0])


def test_convention_unnormalized_proportions_are_rescaled():
    parts = index_splitter(10, [4, 1])
    assert [len(p) for p in parts] == [8, 2]


def test_convention_same_indices_as_torch_random_split_with_manual_seed():
    parts = index_splitter(100, [.8, .2], seed=13)
    torch.manual_seed(13)
    ref = random_split(torch.arange(100), [.8, .2])
    assert _indices(parts[0]) == _indices(ref[0])
    assert [int(i) for i in parts[0].indices] == [int(i) for i in ref[0].indices]
