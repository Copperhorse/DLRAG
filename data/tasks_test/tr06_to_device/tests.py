import torch, torch.nn as nn
from common import bare, make_model
from solution import StepByStep


def test_behavior_valid_device_is_used_and_remembered():
    obj = bare(StepByStep, model=make_model(), device="cuda")
    obj.to("cpu")
    assert obj.device == "cpu" and all(p.device.type == "cpu" for p in obj.model.parameters())


def test_behavior_unknown_device_falls_back_instead_of_crashing():
    obj = bare(StepByStep, model=make_model(), device="cpu")
    obj.to("banana")
    assert all(p.device.type == "cpu" for p in obj.model.parameters())


def test_convention_fallback_is_recorded_and_announced(capsys):
    obj = bare(StepByStep, model=make_model(), device="cpu")
    obj.to("banana")
    out = capsys.readouterr().out
    assert obj.device == "cpu" and "banana" in out and "cpu" in out
