import torch
from torch.utils.data import random_split
from solution import three_way_split


def _idx(s):
    return [int(i) for i in (s.indices if hasattr(s, "indices") else s)]


def test_behavior_three_disjoint_parts_covering_everything_with_the_right_sizes():
    a, b, c = three_way_split(100)
    assert [len(a), len(b), len(c)] == [70, 15, 15]
    assert sorted(_idx(a) + _idx(b) + _idx(c)) == list(range(100))


def test_behavior_reproducible_and_seed_dependent():
    assert _idx(three_way_split(60, seed=2)[0]) == _idx(three_way_split(60, seed=2)[0])
    assert _idx(three_way_split(60, seed=2)[0]) != _idx(three_way_split(60, seed=3)[0])


def test_convention_same_indices_as_torch_random_split_after_manual_seed_with_normalized_proportions():
    for splits, frac in (((70, 15, 15), [.7, .15, .15]), ((3, 1, 1), [.6, .2, .2])):
        got = three_way_split(100, splits, seed=13)
        torch.manual_seed(13)
        ref = random_split(torch.arange(100), frac)
        assert [_idx(g) for g in got] == [_idx(r) for r in ref]
