import torch, torch.nn as nn
from solution import Encoder


def test_behavior_output_is_the_whole_sequence_of_hidden_states():
    enc = Encoder(2, 5)
    out = enc(torch.randn(3, 4, 2))
    assert tuple(out.shape) == (3, 4, 5)


def test_convention_final_hidden_state_is_kept_in_attribute_hidden_sequence_first():
    enc = Encoder(2, 5)
    out = enc(torch.randn(3, 4, 2))
    assert tuple(enc.hidden.shape) == (1, 3, 5)
    assert torch.allclose(enc.hidden[0], out[:, -1], atol=1e-6)


def test_convention_attributes_and_a_batch_first_gru_named_basic_rnn():
    enc = Encoder(2, 5)
    assert enc.n_features == 2 and enc.hidden_dim == 5
    assert isinstance(enc.basic_rnn, nn.GRU) and enc.basic_rnn.batch_first and enc.basic_rnn.num_layers == 1
    assert enc.hidden is None
