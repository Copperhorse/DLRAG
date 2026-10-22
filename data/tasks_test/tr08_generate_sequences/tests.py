import numpy as np
from solution import generate_sequences


def test_behavior_shapes_lengths_labels_and_closeness_to_corners():
    pts, dirs = generate_sequences(50)
    assert len(pts) == 50 and len(dirs) == 50 and set(np.asarray(dirs).tolist()) <= {0, 1}
    assert all(p.shape == (4, 2) for p in pts)
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    assert all(np.min(np.linalg.norm(corners - pt, axis=1)) < 0.6 for p in pts for pt in p)
    pts2, _ = generate_sequences(50, variable_len=True)
    assert {p.shape[0] for p in pts2} <= {2, 3, 4} and len({p.shape[0] for p in pts2}) > 1


def test_behavior_reproducible_for_the_same_seed():
    a, da = generate_sequences(20, seed=3); b, db = generate_sequences(20, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and np.array_equal(da, db)


def test_convention_identical_to_the_books_data_for_the_default_seed():
    def book(n, variable_len, seed):
        corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        np.random.seed(seed)
        bases = np.random.randint(4, size=n)
        lengths = (np.random.randint(3, size=n) + 2) if variable_len else [4] * n
        d = np.random.randint(2, size=n)
        return [corners[[(b + i) % 4 for i in range(4)]][slice(None, None, k * 2 - 1)][:l] + np.random.randn(l, 2) * 0.1
                for b, k, l in zip(bases, d, lengths)], d
    for vl in (False, True):
        got, gd = generate_sequences(30, vl); ref, rd = book(30, vl, 13)
        assert all(np.allclose(x, y) for x, y in zip(got, ref)) and np.array_equal(gd, rd)
