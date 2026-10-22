import numpy as np
from solution import make_lr_fn


def test_behavior_reaches_end_lr_after_num_iter_for_both_modes():
    for mode in ("exp", "linear"):
        fn = make_lr_fn(1e-3, 1.0, 100, mode)
        assert abs(1e-3 * fn(100) - 1.0) < 1e-9
        assert fn(10) < fn(50) < fn(100)


def test_convention_returns_multiplier_that_is_one_at_iteration_zero():
    assert abs(make_lr_fn(1e-3, 1.0, 100)(0) - 1.0) < 1e-12
    assert abs(make_lr_fn(1e-3, 1.0, 100, "linear")(0) - 1.0) < 1e-12


def test_convention_exp_is_geometric_and_linear_is_arithmetic():
    e = make_lr_fn(1e-3, 1.0, 10, "exp")
    assert abs(e(5) - np.sqrt(e(10))) < 1e-9 and abs(e(2) * e(3) - e(5)) < 1e-9
    l = make_lr_fn(1e-3, 1.0, 10, "linear")
    assert abs((l(5) - l(4)) - (l(6) - l(5))) < 1e-9


def test_convention_default_mode_is_exponential():
    assert abs(make_lr_fn(1e-3, 1.0, 10)(5) - make_lr_fn(1e-3, 1.0, 10, "exp")(5)) < 1e-12
