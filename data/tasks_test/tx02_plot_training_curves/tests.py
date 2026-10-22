import torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from stepbystep.v4 import StepByStep
from common import make_model
from solution import plot_training_curves


def _sbs():
    model = make_model()
    sbs = StepByStep(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.1))
    sbs.losses, sbs.val_losses = [4.0, 2.0, 1.0, 0.5], [5.0, 3.0, 2.0, 1.5]
    return sbs


def test_behavior_returns_a_figure_with_both_curves():
    fig = plot_training_curves(_sbs())
    assert isinstance(fig, Figure)
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2
    assert list(lines[0].get_ydata()) == [4.0, 2.0, 1.0, 0.5] and list(lines[1].get_ydata()) == [5.0, 3.0, 2.0, 1.5]
    plt.close(fig)


def test_convention_labels_log_scale_and_axis_names_as_in_the_books_plot():
    fig = plot_training_curves(_sbs()); ax = fig.axes[0]
    assert [l.get_label() for l in ax.get_lines()] == ["Training Loss", "Validation Loss"]
    assert ax.get_yscale() == "log" and ax.get_xlabel() == "Epochs" and ax.get_ylabel() == "Loss"
    assert ax.get_legend() is not None
    plt.close(fig)
