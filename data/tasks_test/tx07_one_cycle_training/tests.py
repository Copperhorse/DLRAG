from torch.optim.lr_scheduler import OneCycleLR
from common import make_model, make_loader
from solution import train_one_cycle


def _run():
    return train_one_cycle(make_model(), make_loader(batch_size=16, n=64, shuffle=True), make_loader(seed=1), 3, 0.2)


def test_behavior_trained_for_the_requested_epochs_and_the_rate_peaks_at_max_lr():
    out = _run()
    lrs = [r[0] if isinstance(r, (list, tuple)) else r for r in out.learning_rates]
    assert out.total_epochs == 3 and abs(max(lrs) - 0.2) < 0.02
    assert lrs.index(max(lrs)) not in (0, len(lrs) - 1)


def test_convention_scheduler_stepped_once_per_mini_batch_and_flagged_as_batch_level():
    out = _run()
    assert isinstance(out.scheduler, OneCycleLR) and out.is_batch_lr_scheduler is True
    assert len(out.learning_rates) == 3 * 4
