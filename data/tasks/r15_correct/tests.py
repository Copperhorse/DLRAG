import torch, torch.nn as nn
from common import bare
from solution import StepByStep


class Fixed(nn.Module):
    def __init__(self, out):
        super().__init__(); self.out = out; self.modes = []
    def forward(self, x):
        self.modes.append(self.training); return self.out


def test_behavior_multiclass_counts_per_class():
    logits = torch.tensor([[5., 0, 0], [0, 5., 0], [0, 0, 5.], [5., 0, 0]])
    obj = bare(StepByStep, model=Fixed(logits), device="cpu")
    out = obj.correct(torch.zeros(4, 1), torch.tensor([0, 1, 1, 0]))
    assert out.tolist() == [[2, 2], [1, 2], [0, 0]]


def test_convention_binary_logits_use_sigmoid_and_two_rows():
    logits = torch.tensor([[2.], [-2.], [-1.], [3.]])
    obj = bare(StepByStep, model=Fixed(logits), device="cpu")
    out = obj.correct(torch.zeros(4, 1), torch.tensor([1, 0, 1, 1]))
    assert out.tolist() == [[1, 1], [2, 3]]


def test_convention_sigmoid_ending_sequential_outputs_are_probabilities():
    class P(nn.Sequential):
        def forward(self, x): return torch.tensor([[0.7], [0.2]])
    model = P(nn.Linear(1, 1), nn.Sigmoid())
    obj = bare(StepByStep, model=model, device="cpu")
    out = obj.correct(torch.zeros(2, 1), torch.tensor([1, 0]))
    assert out.tolist() == [[1, 1], [1, 1]]


def test_convention_threshold_is_respected():
    probs_model = Fixed(torch.tensor([[0.6], [0.6]]))
    obj = bare(StepByStep, model=nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()), device="cpu")
    obj.model.forward = lambda x: torch.tensor([[0.6], [0.6]])
    out = obj.correct(torch.zeros(2, 1), torch.tensor([1, 1]), threshold=0.7)
    assert out.tolist() == [[0, 0], [0, 2]]


def test_convention_eval_during_and_train_after():
    m = Fixed(torch.tensor([[1.0, 0.0]]))
    m.train()
    bare(StepByStep, model=m, device="cpu").correct(torch.zeros(1, 1), torch.tensor([0]))
    assert m.modes == [False] and m.training
