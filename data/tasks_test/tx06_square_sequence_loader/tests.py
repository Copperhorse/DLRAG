import numpy as np, torch
from torch.utils.data import DataLoader
from data_generation.square_sequences import generate_sequences
from solution import square_sequence_loader


def test_behavior_shapes_dtypes_and_batch_size():
    loader = square_sequence_loader(40, batch_size=16)
    xs = torch.cat([x for x, _ in loader]); ys = torch.cat([y for _, y in loader])
    assert isinstance(loader, DataLoader) and loader.batch_size == 16
    assert tuple(xs.shape) == (40, 4, 2) and tuple(ys.shape) == (40, 1)
    assert xs.dtype == torch.float32 and ys.dtype == torch.float32


def test_convention_samples_are_the_books_generator_output_in_generated_order():
    pts, d = generate_sequences(n=40)
    loader = square_sequence_loader(40)
    xs = torch.cat([x for x, _ in loader]); ys = torch.cat([y for _, y in loader])
    assert torch.allclose(xs, torch.as_tensor(np.array(pts)).float(), atol=1e-6)
    assert torch.equal(ys, torch.as_tensor(d).view(-1, 1).float())
