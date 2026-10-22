import numpy as np, torch, torch.nn.functional as F
from solution import Attention


def _setup(proj_values=False, seed=0):
    torch.manual_seed(seed)
    att = Attention(6, proj_values=proj_values)
    keys = torch.randn(2, 5, 6)
    att.init_keys(keys)
    return att, keys, torch.randn(2, 1, 6)


def test_behavior_context_shape_and_masked_positions_are_ignored():
    att, keys, q = _setup()
    mask = torch.tensor([[[1, 1, 1, 0, 0]], [[1, 1, 1, 1, 0]]])
    out = att(q, mask=mask)
    assert tuple(out.shape) == (2, 1, 6)
    keys2 = keys.clone(); keys2[0, 3:] += 100.0; keys2[1, 4:] += 100.0      # change only the masked sources
    att.init_keys(keys2)
    assert torch.allclose(att(q, mask=mask), out, atol=1e-4)


def test_convention_parameter_names_of_the_three_projections():
    names = {n for n, _ in Attention(6).named_parameters()}
    assert names == {"linear_query.weight", "linear_query.bias", "linear_key.weight", "linear_key.bias",
                     "linear_value.weight", "linear_value.bias"}


def test_convention_scores_are_scaled_dot_products_of_projected_query_and_keys():
    att, keys, q = _setup()
    expected = torch.bmm(att.linear_query(q), att.linear_key(keys).permute(0, 2, 1)) / np.sqrt(6)
    assert torch.allclose(att.score_function(q), expected, atol=1e-5)


def test_convention_weights_kept_in_alphas_detached_and_values_projected_only_on_request():
    att, keys, q = _setup()
    att(q)
    assert tuple(att.alphas.shape) == (2, 1, 5) and not att.alphas.requires_grad
    assert torch.allclose(att.alphas.sum(-1), torch.ones(2, 1), atol=1e-5)
    assert torch.equal(att.values, keys)                                   # proj_values=False -> raw keys
    att2, keys2, _ = _setup(proj_values=True)
    assert torch.allclose(att2.values, att2.linear_value(keys2), atol=1e-6)
