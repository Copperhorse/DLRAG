import torch, torch.nn as nn, torch.optim as optim
from stepbystep.v4 import StepByStep
from common import make_model, make_loader, snapshot, same_params
from solution import resume_training


def _first_run(path):
    model = make_model(0)
    sbs = StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.1))
    sbs.set_loaders(make_loader(shuffle=True), make_loader(seed=1))
    sbs.train(2)
    sbs.save_checkpoint(path)
    return sbs


def test_behavior_continues_from_the_saved_weights_not_from_the_fresh_model(tmp_path):
    first = _first_run(str(tmp_path / "ck.pth"))
    out = resume_training(str(tmp_path / "ck.pth"), make_model(5), make_loader(shuffle=True), make_loader(seed=1), 1)
    fresh_loss = nn.MSELoss()(make_model(5)(torch.as_tensor(make_loader().dataset.tensors[0])), make_loader().dataset.tensors[1]).item()
    assert out.losses[2] < fresh_loss and abs(out.losses[2] - first.losses[-1]) < abs(out.losses[2] - fresh_loss)


def test_convention_history_contains_the_earlier_epochs_and_the_epoch_counter_continues(tmp_path):
    first = _first_run(str(tmp_path / "ck.pth"))
    out = resume_training(str(tmp_path / "ck.pth"), make_model(5), make_loader(shuffle=True), make_loader(seed=1), 3)
    assert out.total_epochs == 5 and len(out.losses) == 5 and len(out.val_losses) == 5
    assert out.losses[:2] == first.losses and out.val_losses[:2] == first.val_losses


def test_convention_optimizer_state_is_restored_too(tmp_path):
    _first_run(str(tmp_path / "ck.pth"))
    out = resume_training(str(tmp_path / "ck.pth"), make_model(5), make_loader(shuffle=True), make_loader(seed=1), 1)
    assert out.optimizer.state_dict()["param_groups"][0]["lr"] == 0.1 and type(out).__name__ == "StepByStep"
