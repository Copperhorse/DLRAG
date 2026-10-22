import torch
from torch.utils.data import WeightedRandomSampler
from solution import make_balanced_sampler


def _labels():
    return torch.cat([torch.zeros(90), torch.ones(10)]).long().view(-1, 1)


def test_behavior_draws_each_class_about_equally():
    y = _labels()
    s = make_balanced_sampler(y)
    if hasattr(s, "generator") and s.generator is not None:
        s.generator.manual_seed(0)
    drawn = torch.tensor(list(s))
    assert len(drawn) == len(y)
    frac_minority = (y.squeeze()[drawn] == 1).float().mean().item()
    assert 0.35 < frac_minority < 0.65


def test_convention_is_weighted_random_sampler_with_replacement():
    s = make_balanced_sampler(_labels())
    assert isinstance(s, WeightedRandomSampler)
    assert s.replacement is True and s.num_samples == 100


def test_convention_weights_are_inverse_class_frequency_per_sample():
    s = make_balanced_sampler(_labels())
    w = torch.as_tensor(s.weights).float()
    assert torch.allclose(w[:90], torch.full((90,), 1 / 90.0)) and torch.allclose(w[90:], torch.full((10,), 1 / 10.0))


def test_convention_exposes_a_torch_generator_attribute_for_seeding():
    s = make_balanced_sampler(_labels())
    assert isinstance(s.generator, torch.Generator)
