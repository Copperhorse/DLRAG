import torch.optim as optim
from torch.optim import lr_scheduler as S
from common import bare, make_model
from solution import StepByStep


def _obj():
    model = make_model()
    return bare(StepByStep, model=model, optimizer=optim.SGD(model.parameters(), lr=0.1),
                scheduler=None, is_batch_lr_scheduler=False)


def test_behavior_accepts_scheduler_of_own_optimizer():
    obj = _obj()
    sch = S.StepLR(obj.optimizer, step_size=2)
    obj.set_lr_scheduler(sch)
    assert obj.scheduler is sch


def test_convention_ignores_scheduler_of_another_optimizer():
    obj = _obj()
    other = optim.SGD(make_model().parameters(), lr=0.1)
    obj.set_lr_scheduler(S.StepLR(other, step_size=2))
    assert obj.scheduler is None


def test_convention_epoch_level_schedulers_are_not_batch_level():
    for make in (lambda o: S.StepLR(o, step_size=2), lambda o: S.ReduceLROnPlateau(o), lambda o: S.ExponentialLR(o, gamma=0.9)):
        obj = _obj()
        obj.set_lr_scheduler(make(obj.optimizer))
        assert obj.is_batch_lr_scheduler is False


def test_convention_batch_level_schedulers():
    makers = (lambda o: S.CyclicLR(o, base_lr=0.01, max_lr=0.1, step_size_up=5),
              lambda o: S.OneCycleLR(o, max_lr=0.1, total_steps=10),
              lambda o: S.CosineAnnealingWarmRestarts(o, T_0=5))
    for make in makers:
        obj = _obj()
        obj.set_lr_scheduler(make(obj.optimizer))
        assert obj.is_batch_lr_scheduler is True
