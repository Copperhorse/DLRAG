import numpy as np
from solution import make_data


def _book():
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + (.1 * np.random.randn(100, 1))
    idx = np.arange(100)
    np.random.shuffle(idx)
    return x[idx[:80]], y[idx[:80]], x[idx[80:]], y[idx[80:]]


def test_behavior_shapes_split_and_linear_relation():
    xt, yt, xv, yv = make_data()
    assert xt.shape == (80, 1) and yt.shape == (80, 1) and xv.shape == (20, 1) and yv.shape == (20, 1)
    w, b = np.polyfit(xt[:, 0], yt[:, 0], 1)
    assert abs(w - 2) < 0.2 and abs(b - 1) < 0.2


def test_behavior_reproducible():
    a, b = make_data(), make_data()
    assert all(np.array_equal(p, q) for p, q in zip(a, b))


def test_convention_identical_to_the_books_data():
    got, ref = make_data(), _book()
    assert all(np.array_equal(p, q) for p, q in zip(got, ref))
