import torch.optim as optim
from torch.optim import lr_scheduler as S
from common import bare, make_model
from solution import StepByStep


def _obj(make, batch=True, total_epochs=1):
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=1.0)
    return bare(StepByStep, model=model, optimizer=opt, scheduler=make(opt) if make else None,
                is_batch_lr_scheduler=batch, total_epochs=total_epochs, learning_rates=[])


def test_behavior_steps_batch_level_scheduler():
    obj = _obj(lambda o: S.CyclicLR(o, base_lr=0.1, max_lr=1.0, step_size_up=4))
    start = obj.optimizer.param_groups[0]["lr"]
    obj._mini_batch_schedulers(0.25)
    assert obj.optimizer.param_groups[0]["lr"] != start


def test_convention_records_list_of_lrs_per_call():
    obj = _obj(lambda o: S.CyclicLR(o, base_lr=0.1, max_lr=1.0, step_size_up=4))
    obj._mini_batch_schedulers(0.0); obj._mini_batch_schedulers(0.5)
    assert len(obj.learning_rates) == 2 and all(isinstance(r, list) and len(r) == 1 for r in obj.learning_rates)


def test_convention_warm_restarts_scheduler_gets_fractional_epoch_position():
    obj = _obj(lambda o: S.CosineAnnealingWarmRestarts(o, T_0=4), total_epochs=2)
    obj._mini_batch_schedulers(0.5)
    ref_opt = optim.SGD(make_model().parameters(), lr=1.0)
    ref = S.CosineAnnealingWarmRestarts(ref_opt, T_0=4)
    ref.step(2 + 0.5)
    assert abs(obj.optimizer.param_groups[0]["lr"] - ref_opt.param_groups[0]["lr"]) < 1e-9


def test_convention_epoch_level_scheduler_is_left_alone():
    obj = _obj(lambda o: S.StepLR(o, step_size=1, gamma=0.5), batch=False)
    obj._mini_batch_schedulers(0.5)
    assert obj.optimizer.param_groups[0]["lr"] == 1.0 and obj.learning_rates == []
