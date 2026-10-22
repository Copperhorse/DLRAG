import torch, torch.nn as nn
from solution import BiEncoder


def test_behavior_outputs_have_both_directions_concatenated():
    enc = BiEncoder(2, 5)
    out = enc(torch.randn(3, 4, 2))
    assert tuple(out.shape) == (3, 4, 10)


def test_convention_hidden_attribute_holds_both_final_states_sequence_first():
    enc = BiEncoder(2, 5)
    out = enc(torch.randn(3, 4, 2))
    assert tuple(enc.hidden.shape) == (2, 3, 5)
    assert torch.allclose(enc.hidden[0], out[:, -1, :5], atol=1e-6) and torch.allclose(enc.hidden[1], out[:, 0, 5:], atol=1e-6)


def test_convention_attribute_names_and_the_gru_called_basic_rnn():
    enc = BiEncoder(2, 5)
    assert enc.n_features == 2 and enc.hidden_dim == 5 and enc.hidden is None
    assert isinstance(enc.basic_rnn, nn.GRU) and enc.basic_rnn.bidirectional and enc.basic_rnn.batch_first
