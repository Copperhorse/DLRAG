import numpy as np, torch, torch.nn as nn
from common import bare
from solution import StepByStep


class Probe(nn.Module):
    def __init__(self):
        super().__init__(); self.lin = nn.Linear(2, 1); self.modes = []
    def forward(self, x):
        self.modes.append(self.training); return self.lin(x)


def _obj():
    torch.manual_seed(0)
    return bare(StepByStep, model=Probe(), device="cpu")


def test_behavior_returns_probabilities_as_numpy():
    obj = _obj()
    x = np.random.RandomState(0).randn(6, 2)
    out = obj.predict_proba(x)
    expected = torch.sigmoid(obj.model.lin(torch.as_tensor(x).float())).detach().numpy()
    assert isinstance(out, np.ndarray) and out.shape == (6, 1) and np.allclose(out, expected, atol=1e-6)
    assert ((out >= 0) & (out <= 1)).all()


def test_convention_float64_input_is_cast_to_float32():
    out = _obj().predict_proba(np.zeros((3, 2), dtype=np.float64))
    assert out.dtype == np.float32


def test_convention_eval_while_predicting_and_train_afterwards():
    obj = _obj()
    obj.predict_proba(np.zeros((2, 2)))
    assert obj.model.modes == [False] and obj.model.training
