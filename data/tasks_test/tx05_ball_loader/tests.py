import torch
from torch.utils.data import DataLoader
from data_generation.ball import load_data
from solution import ball_loader


def test_behavior_a_single_batch_with_float_features_and_binary_labels():
    loader = ball_loader(200, 5)
    batches = list(loader)
    assert isinstance(loader, DataLoader) and len(batches) == 1
    x, y = batches[0]
    assert tuple(x.shape) == (200, 5) and tuple(y.shape) == (200, 1)
    assert x.dtype == torch.float32 and y.dtype == torch.float32 and set(y.unique().tolist()) <= {0.0, 1.0}


def test_convention_data_are_the_books_generator_output_with_its_default_seed_and_order():
    X, y = load_data(n_dims=5, n_points=200)
    x_got, y_got = next(iter(ball_loader(200, 5)))
    assert torch.allclose(x_got, torch.as_tensor(X).float(), atol=1e-6) and torch.equal(y_got, torch.as_tensor(y).float())
