import numpy as np, torch
from solution import SimplePositionalEncoding


def _table(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(0, max_len).float().unsqueeze(1)
    speed = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * speed); pe[:, 1::2] = torch.cos(pos * speed)
    return pe


def test_behavior_adds_the_table_to_an_unscaled_input():
    m = SimplePositionalEncoding(30, 8)
    x = torch.randn(2, 7, 8)
    assert torch.allclose(m(x), x + _table(30, 8)[:7].unsqueeze(0), atol=1e-5)
    assert len(list(m.parameters())) == 0


def test_convention_table_is_a_buffer_named_pe_shaped_1_maxlen_dmodel():
    m = SimplePositionalEncoding(30, 8)
    assert "pe" in dict(m.named_buffers()) and tuple(m.pe.shape) == (1, 30, 8)
    assert "pe" in m.state_dict()
