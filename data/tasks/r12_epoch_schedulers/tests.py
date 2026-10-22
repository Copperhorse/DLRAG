import torch.optim as optim
from torch.optim import lr_scheduler as S
from common import bare, make_model
from solution import StepByStep


def _obj(make, batch=False):
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=1.0)
    sch = make(opt) if make else None
    return bare(StepByStep, model=model, optimizer=opt, scheduler=sch,
                is_batch_lr_scheduler=batch, learning_rates=[])


def test_behavior_steps_epoch_scheduler_each_call():
    obj = _obj(lambda o: S.StepLR(o, step_size=1, gamma=0.5))
    for _ in range(3):
        obj._epoch_schedulers(0.1)
    assert abs(obj.optimizer.param_groups[0]["lr"] - 0.125) < 1e-9


def test_convention_records_a_list_of_lrs_per_call():
    obj = _obj(lambda o: S.StepLR(o, step_size=1, gamma=0.5))
    obj._epoch_schedulers(0.1); obj._epoch_schedulers(0.1)
    assert obj.learning_rates == [[0.5], [0.25]]


def test_convention_reduce_on_plateau_receives_validation_loss():
    obj = _obj(lambda o: S.ReduceLROnPlateau(o, patience=0, factor=0.1))
    obj._epoch_schedulers(1.0)   # first value: best so far
    obj._epoch_schedulers(2.0)   # worse -> lr reduced
    assert abs(obj.optimizer.param_groups[0]["lr"] - 0.1) < 1e-9
    assert len(obj.learning_rates) == 2


def test_convention_batch_level_scheduler_is_left_alone():
    obj = _obj(lambda o: S.StepLR(o, step_size=1, gamma=0.5), batch=True)
    obj._epoch_schedulers(0.1)
    assert obj.optimizer.param_groups[0]["lr"] == 1.0 and obj.learning_rates == []


def test_convention_no_scheduler_is_a_noop():
    obj = _obj(None)
    obj._epoch_schedulers(0.1)
    assert obj.learning_rates == []
