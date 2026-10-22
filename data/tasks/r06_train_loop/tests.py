import torch
from common import bare
from solution import StepByStep


def _make(val=True):
    log = {"seed": [], "calls": [], "grad_enabled": []}
    class Sub(StepByStep):
        def set_seed(self, seed=42):
            log["seed"].append(seed)
        def _mini_batch(self, validation=False):
            log["calls"].append(validation)
            log["grad_enabled"].append(torch.is_grad_enabled())
            if validation:
                return 0.5 if val else None
            return float(len([c for c in log["calls"] if not c]))
    return bare(Sub, losses=[], val_losses=[], total_epochs=0), log


def test_behavior_runs_requested_epochs_and_can_resume():
    obj, log = _make()
    obj.train(3)
    obj.train(2)
    assert log["calls"].count(False) == 5


def test_convention_tracks_losses_val_losses_and_total_epochs_attributes():
    obj, _ = _make()
    obj.train(3)
    assert obj.total_epochs == 3
    assert obj.losses == [1.0, 2.0, 3.0]
    assert obj.val_losses == [0.5, 0.5, 0.5]


def test_convention_seeds_once_before_training_with_given_seed():
    obj, log = _make()
    obj.train(2, seed=7)
    assert log["seed"] == [7]


def test_convention_validation_runs_without_gradients():
    obj, log = _make()
    obj.train(2)
    val_flags = [g for c, g in zip(log["calls"], log["grad_enabled"]) if c]
    assert val_flags and not any(val_flags)
