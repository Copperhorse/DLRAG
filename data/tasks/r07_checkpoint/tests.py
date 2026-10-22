import torch, torch.optim as optim
from common import bare, make_model, snapshot, same_params
from solution import StepByStep


def _obj(seed=0, epochs=3):
    model = make_model(seed)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return bare(StepByStep, model=model, optimizer=opt, losses=[3.0, 2.0, 1.0][:epochs],
                val_losses=[3.5, 2.5, 1.5][:epochs], total_epochs=epochs)


def test_behavior_round_trip_restores_model_and_optimizer(tmp_path):
    a = _obj(0)
    loss = a.model(torch.ones(4, 1)).sum()
    loss.backward(); a.optimizer.step()
    path = str(tmp_path / "ck.pth")
    a.save_checkpoint(path)
    b = _obj(1)
    b.load_checkpoint(path)
    assert same_params(snapshot(a.model), snapshot(b.model))
    assert b.optimizer.state_dict()["state"].keys() == a.optimizer.state_dict()["state"].keys()


def test_convention_checkpoint_file_has_the_books_keys(tmp_path):
    a = _obj()
    path = str(tmp_path / "ck.pth")
    a.save_checkpoint(path)
    ck = torch.load(path, weights_only=False)
    assert set(ck) == {"epoch", "model_state_dict", "optimizer_state_dict", "loss", "val_loss"}
    assert ck["epoch"] == 3 and ck["loss"] == [3.0, 2.0, 1.0] and ck["val_loss"] == [3.5, 2.5, 1.5]


def test_convention_load_restores_history_attributes_and_train_mode(tmp_path):
    a = _obj()
    path = str(tmp_path / "ck.pth")
    a.save_checkpoint(path)
    b = _obj(1, epochs=1)
    b.model.eval()
    b.load_checkpoint(path)
    assert b.total_epochs == 3 and b.losses == [3.0, 2.0, 1.0] and b.val_losses == [3.5, 2.5, 1.5]
    assert b.model.training
