import torch, torch.optim as optim
from torch.optim import lr_scheduler as S
from common import bare, make_model
from solution import StepByStep


def _obj(with_sched=True, seed=0):
    model = make_model(seed)
    opt = optim.SGD(model.parameters(), lr=1.0)
    sch = S.StepLR(opt, step_size=1, gamma=0.5) if with_sched else None
    return bare(StepByStep, model=model, optimizer=opt, scheduler=sch,
                losses=[2.0, 1.0], val_losses=[2.5, 1.5], total_epochs=2)


def test_behavior_scheduler_state_round_trips(tmp_path):
    a = _obj()
    a.scheduler.step(); a.scheduler.step()
    path = str(tmp_path / "ck.pth"); a.save_checkpoint(path)
    b = _obj(seed=1)
    b.load_checkpoint(path)
    assert b.scheduler.last_epoch == 2
    assert abs(b.optimizer.param_groups[0]["lr"] - 0.25) < 1e-9


def test_convention_keeps_the_books_five_keys_and_adds_scheduler_key(tmp_path):
    path = str(tmp_path / "ck.pth"); _obj().save_checkpoint(path)
    ck = torch.load(path, weights_only=False)
    assert set(ck) == {"epoch", "model_state_dict", "optimizer_state_dict", "loss", "val_loss", "scheduler_state_dict"}


def test_convention_no_scheduler_means_no_extra_key_and_loading_old_checkpoints_works(tmp_path):
    path = str(tmp_path / "ck.pth"); _obj(with_sched=False).save_checkpoint(path)
    assert "scheduler_state_dict" not in torch.load(path, weights_only=False)
    b = _obj(with_sched=True)
    b.load_checkpoint(path)                      # scheduler present on the object, key absent in file
    assert b.total_epochs == 2 and b.losses == [2.0, 1.0] and b.val_losses == [2.5, 1.5]
    assert b.model.training
