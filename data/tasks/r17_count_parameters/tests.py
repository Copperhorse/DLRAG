import torch.nn as nn
from common import bare
from solution import StepByStep


def test_behavior_counts_all_parameters_of_a_plain_model():
    obj = bare(StepByStep, model=nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1)))
    assert obj.count_parameters() == 3 * 4 + 4 + 4 * 1 + 1


def test_convention_frozen_parameters_are_not_counted():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
    for p in model[0].parameters():
        p.requires_grad = False
    assert bare(StepByStep, model=model).count_parameters() == 4 * 1 + 1


def test_convention_returns_int():
    assert type(bare(StepByStep, model=nn.Linear(2, 2)).count_parameters()) is int
