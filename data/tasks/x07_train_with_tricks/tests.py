import torch, torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from common import make_model, make_loader
from solution import train_with_tricks


def _run(n=6, step=2):
    return train_with_tricks(make_model(), make_loader(shuffle=True), make_loader(seed=1), n, 0.5, step)


def test_behavior_trained_for_the_requested_epochs_and_learning_rate_decays():
    out = _run()
    assert out.total_epochs == 6 and len(out.losses) == 6
    lrs = [r[0] if isinstance(r, (list, tuple)) else r for r in out.learning_rates]
    assert lrs[-1] < lrs[0]


def test_convention_gradient_clipping_is_configured_on_the_object():
    out = _run()
    assert callable(out.clipping)


def test_convention_epoch_level_step_scheduler_recorded_per_epoch():
    out = _run(n=6, step=2)
    assert isinstance(out.scheduler, StepLR) and out.is_batch_lr_scheduler is False
    assert out.learning_rates == [[0.1], [0.05], [0.05], [0.025], [0.025], [0.0125]]
