import numpy as np
from common import make_loader
from solution import mini_batch


def test_behavior_returns_mean_of_batch_losses():
    seen = []
    def step(x, y):
        seen.append(len(x))
        return float(len(seen))
    loss = mini_batch("cpu", make_loader(batch_size=16, n=64), step)
    assert len(seen) == 4
    assert abs(float(loss) - 2.5) < 1e-9


def test_convention_moves_batches_to_device_before_step():
    devices = []
    def step(x, y):
        devices.append((x.device.type, y.device.type))
        return 0.0
    mini_batch("meta", make_loader(), step)
    assert devices and all(d == ("meta", "meta") for d in devices)


def test_convention_unequal_last_batch_is_plain_mean_of_batch_losses():
    vals = iter([1.0, 2.0, 6.0])
    loss = mini_batch("cpu", make_loader(batch_size=30, n=70), lambda x, y: next(vals))
    assert abs(float(loss) - 3.0) < 1e-9
