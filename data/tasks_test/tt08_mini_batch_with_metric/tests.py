from common import make_loader
from solution import mini_batch_with_metric


def test_behavior_plain_averages_of_loss_and_metric():
    vals = iter([(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (6.0, 60.0)])
    loss, metric = mini_batch_with_metric("cpu", make_loader(batch_size=16, n=64), lambda x, y: next(vals))
    assert abs(float(loss) - 3.0) < 1e-9 and abs(float(metric) - 30.0) < 1e-9


def test_convention_batches_are_moved_to_the_device_before_the_step():
    seen = []
    mini_batch_with_metric("meta", make_loader(), lambda x, y: (seen.append((x.device.type, y.device.type)) or (0.0, 0.0)))
    assert seen and all(s == ("meta", "meta") for s in seen)


def test_convention_unweighted_mean_over_batches_even_when_the_last_batch_is_smaller():
    vals = iter([(1.0, 1.0), (2.0, 2.0), (6.0, 6.0)])
    loss, metric = mini_batch_with_metric("cpu", make_loader(batch_size=30, n=70), lambda x, y: next(vals))
    assert abs(float(loss) - 3.0) < 1e-9 and abs(float(metric) - 3.0) < 1e-9
