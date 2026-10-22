import torch
from solution import subsequent_mask


def test_behavior_lower_triangular_with_diagonal():
    m = subsequent_mask(4)
    assert tuple(m.shape) == (1, 4, 4)
    for i in range(4):
        for j in range(4):
            assert bool(m[0, i, j]) == (j <= i)


def test_behavior_size_one():
    assert tuple(subsequent_mask(1).shape) == (1, 1, 1) and bool(subsequent_mask(1)[0, 0, 0])


def test_convention_is_a_torch_bool_tensor():
    m = subsequent_mask(3)
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool
