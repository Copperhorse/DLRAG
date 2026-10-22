import torch, torch.nn as nn
from common import make_model, make_loader, snapshot, same_params
from solution import train_regression


def _val_loss(model, loader):
    with torch.no_grad():
        return sum(nn.MSELoss()(model(x), y).item() for x, y in loader) / len(loader)


def test_behavior_model_is_trained():
    model = make_model()
    tl, vl = make_loader(shuffle=True), make_loader(seed=1)
    before, loss0 = snapshot(model), _val_loss(model, vl)
    out = train_regression(model, tl, vl, 5)
    assert out is not None and not same_params(before, snapshot(model))
    assert _val_loss(model, vl) < loss0


def test_convention_returns_the_books_training_class_with_its_history():
    out = train_regression(make_model(), make_loader(shuffle=True), make_loader(seed=1), 4)
    assert type(out).__name__ == "StepByStep"
    assert out.total_epochs == 4 and len(out.losses) == 4 and len(out.val_losses) == 4


def test_convention_uses_the_given_loaders_and_the_books_training_loop():
    tl, vl = make_loader(shuffle=True), make_loader(seed=1)
    out = train_regression(make_model(), tl, vl, 2)
    assert out.train_loader is tl and out.val_loader is vl
    assert out.losses[1] < out.losses[0]
