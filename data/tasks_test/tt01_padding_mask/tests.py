import torch
from solution import padding_mask


def test_behavior_true_for_real_positions_false_for_padding():
    m = padding_mask([2, 4, 1], 4)
    expected = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]).bool()
    assert torch.equal(m.reshape(3, 4), expected)


def test_convention_shape_n_1_l_and_bool_dtype_so_it_broadcasts_over_the_scores():
    m = padding_mask(torch.tensor([3, 5]), 6)
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool and tuple(m.shape) == (2, 1, 6)
