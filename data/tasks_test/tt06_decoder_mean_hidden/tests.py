import torch
from common import bare
from solution import Decoder


def test_behavior_state_is_the_average_over_positions_in_gru_layout():
    obj = bare(Decoder)
    seq = torch.randn(3, 5, 4)
    obj.init_hidden_mean(seq)
    h = obj.__dict__.get("hidden")
    assert h is not None and tuple(h.shape) == (1, 3, 4)
    assert torch.allclose(h[0], seq.mean(dim=1), atol=1e-6)


def test_convention_stored_in_attribute_hidden_and_not_the_last_position():
    obj = bare(Decoder)
    seq = torch.zeros(2, 4, 3); seq[:, -1] = 8.0
    obj.init_hidden_mean(seq)
    assert torch.allclose(obj.hidden, torch.full((1, 2, 3), 2.0))
