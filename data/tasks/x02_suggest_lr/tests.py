import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from stepbystep.v4 import StepByStep
from common import make_model, make_loader, snapshot, same_params
from solution import suggest_lr


def _sbs(lr=1e-3):
    model = make_model()
    return StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=lr))


def _expected(sbs, loader, end_lr, num_iter):
    tracking, fig = sbs.lr_range_test(loader, end_lr=end_lr, num_iter=num_iter)
    plt.close(fig)
    return float(tracking["lr"][int(np.argmin(tracking["loss"]))])


def test_behavior_returns_a_learning_rate_inside_the_tested_range():
    lr = suggest_lr(_sbs(), make_loader(), end_lr=1.0, num_iter=30)
    assert isinstance(lr, float) and 1e-3 <= lr <= 1.0


def test_behavior_model_and_optimizer_are_left_unchanged():
    sbs = _sbs(); before = snapshot(sbs.model)
    suggest_lr(sbs, make_loader(), end_lr=1.0, num_iter=30)
    assert same_params(before, snapshot(sbs.model))
    assert sbs.optimizer.param_groups[0]["lr"] == 1e-3


def test_convention_value_comes_from_the_books_range_test_schedule():
    loader = make_loader()
    got = suggest_lr(_sbs(), loader, end_lr=0.5, num_iter=25)
    assert abs(got - _expected(_sbs(), loader, 0.5, 25)) < 1e-12
