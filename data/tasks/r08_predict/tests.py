import numpy as np, torch, torch.nn as nn
from common import bare, make_model
from solution import StepByStep


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.modes = []
    def forward(self, x):
        self.modes.append(self.training)
        return self.lin(x)


def _obj():
    torch.manual_seed(0)
    m = Probe()
    m.train()
    return bare(StepByStep, model=m, device="cpu")


def test_behavior_returns_numpy_predictions():
    obj = _obj()
    x = np.random.RandomState(0).rand(5, 1)
    out = obj.predict(x)
    assert isinstance(out, np.ndarray) and out.shape == (5, 1)
    expected = obj.model.lin(torch.as_tensor(x).float()).detach().numpy()
    assert np.allclose(out, expected, atol=1e-6)


def test_convention_float_input_even_for_float64_arrays():
    obj = _obj()
    obj.predict(np.zeros((3, 1), dtype=np.float64))   # would raise a dtype error without .float()


def test_convention_uses_eval_mode_while_predicting():
    obj = _obj()
    obj.predict(np.zeros((2, 1)))
    assert obj.model.modes == [False]


def test_convention_restores_train_mode_afterwards():
    obj = _obj()
    obj.predict(np.zeros((2, 1)))
    assert obj.model.training


def test_convention_output_is_detached():
    obj = _obj()
    out = obj.predict(np.zeros((2, 1)))
    assert not isinstance(out, torch.Tensor)
