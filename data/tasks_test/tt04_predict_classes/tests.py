import numpy as np, torch, torch.nn as nn
from common import bare
from solution import StepByStep


class Fixed(nn.Module):
    def __init__(self, logits):
        super().__init__(); self.logits = logits; self.modes = []
    def forward(self, x):
        self.modes.append(self.training); return self.logits


def _obj(logits):
    m = Fixed(torch.tensor(logits).view(-1, 1)); m.train()
    return bare(StepByStep, model=m, device="cpu")


def test_behavior_classes_from_logits():
    out = _obj([2.0, -2.0, 0.5, -0.5]).predict_classes(np.zeros((4, 2)))
    assert isinstance(out, np.ndarray) and out.shape == (4, 1) and out.ravel().tolist() == [1, 0, 1, 0]


def test_convention_threshold_is_strict_and_applied_to_probabilities():
    assert _obj([0.0]).predict_classes(np.zeros((1, 2))).ravel().tolist() == [0]            # sigmoid(0) = .5 is not > .5
    assert _obj([1.0]).predict_classes(np.zeros((1, 2)), threshold=0.8).ravel().tolist() == [0]   # sigmoid(1) = .73
    assert _obj([1.0]).predict_classes(np.zeros((1, 2)), threshold=0.7).ravel().tolist() == [1]


def test_convention_eval_while_predicting_train_afterwards_and_integer_output():
    obj = _obj([1.0, -1.0])
    out = obj.predict_classes(np.zeros((2, 2), dtype=np.float64))
    assert obj.model.modes == [False] and obj.model.training and np.issubdtype(out.dtype, np.integer)
