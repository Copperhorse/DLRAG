import torch
from common import bare
from solution import StepByStep


def _make(val_losses):
    log = {"seed": [], "grad": [], "val_i": 0}
    class Sub(StepByStep):
        def set_seed(self, seed=42):
            log["seed"].append(seed)
        def _mini_batch(self, validation=False):
            if validation:
                log["grad"].append(torch.is_grad_enabled())
                v = val_losses[min(log["val_i"], len(val_losses) - 1)]
                log["val_i"] += 1
                return v
            return 1.0
    return bare(Sub, losses=[], val_losses=[], total_epochs=0), log


def test_behavior_without_patience_runs_all_epochs():
    obj, _ = _make([1.0, 2.0, 3.0, 4.0])
    obj.train(4)
    assert obj.total_epochs == 4 and len(obj.losses) == 4 and len(obj.val_losses) == 4


def test_behavior_stops_after_patience_epochs_without_improvement():
    obj, _ = _make([5.0, 4.0, 4.0, 4.5, 3.9, 9.0])
    obj.train(10, patience=2)
    assert obj.total_epochs == 4 and obj.val_losses == [5.0, 4.0, 4.0, 4.5]


def test_convention_equal_value_is_not_an_improvement():
    obj, _ = _make([2.0, 2.0, 2.0, 2.0])
    obj.train(10, patience=1)
    assert obj.total_epochs == 2


def test_convention_book_style_bookkeeping():
    obj, log = _make([3.0, 2.0, 1.0])
    obj.train(3, seed=11)
    assert log["seed"] == [11] and obj.losses == [1.0, 1.0, 1.0] and obj.val_losses == [3.0, 2.0, 1.0]
    assert not any(log["grad"])


def test_convention_resuming_keeps_counting_total_epochs():
    obj, _ = _make([3.0, 2.0, 1.0, 0.5])
    obj.train(2); obj.train(2)
    assert obj.total_epochs == 4
