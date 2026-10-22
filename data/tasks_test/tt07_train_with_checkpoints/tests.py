from common import bare
from solution import StepByStep


def _obj(start=0):
    log = {"train": [], "save": []}
    class Sub(StepByStep):
        def train(self, n_epochs, seed=42):
            log["train"].append((n_epochs, seed)); self.total_epochs += n_epochs
        def save_checkpoint(self, filename):
            log["save"].append(filename)
    return bare(Sub, total_epochs=start), log


def test_behavior_chunks_and_one_checkpoint_per_chunk():
    obj, log = _obj()
    obj.train_with_checkpoints(7, 3, "ck_{epoch}.pth")
    assert [n for n, _ in log["train"]] == [3, 3, 1]
    assert log["save"] == ["ck_3.pth", "ck_6.pth", "ck_7.pth"]


def test_convention_epoch_in_the_name_is_the_total_trained_so_far_including_earlier_training():
    obj, log = _obj(start=10)
    obj.train_with_checkpoints(4, 2, "run/{epoch:03d}.pth")
    assert log["save"] == ["run/012.pth", "run/014.pth"] and obj.total_epochs == 14


def test_convention_seed_is_passed_to_every_train_call_and_no_extra_checkpoint_when_divisible():
    obj, log = _obj()
    obj.train_with_checkpoints(6, 3, "c{epoch}", seed=9)
    assert log["train"] == [(3, 9), (3, 9)] and log["save"] == ["c3", "c6"]
