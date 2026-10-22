import numpy as np, torch
from solution import PositionalEncoding


def _table(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(0, max_len).float().unsqueeze(1)
    speed = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * speed); pe[:, 1::2] = torch.cos(pos * speed)
    return pe


def test_behavior_shape_no_parameters_and_position_dependence():
    m = PositionalEncoding(20, 8)
    x = torch.zeros(2, 5, 8)
    out = m(x)
    assert tuple(out.shape) == (2, 5, 8) and len(list(m.parameters())) == 0
    assert not torch.allclose(out[:, 0], out[:, 1])
    assert torch.allclose(out[0], out[1])


def test_convention_table_is_a_buffer_named_pe_with_a_leading_batch_dimension():
    m = PositionalEncoding(20, 8)
    assert "pe" in dict(m.named_buffers()) and tuple(m.pe.shape) == (1, 20, 8)
    assert torch.allclose(m.pe[0], _table(20, 8), atol=1e-6)


def test_convention_input_is_scaled_by_sqrt_d_model_before_adding():
    m = PositionalEncoding(20, 8)
    x = torch.randn(3, 6, 8)
    expected = x * np.sqrt(8) + _table(20, 8)[:6].unsqueeze(0)
    assert torch.allclose(m(x), expected, atol=1e-5)
