import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Normalize
from solution import image_normalizer


def _data():
    g = torch.Generator().manual_seed(0)
    images = torch.rand(40, 3, 4, 4, generator=g) * torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
    return images, DataLoader(TensorDataset(images, torch.zeros(40)), batch_size=16)   # last batch has 8 images


def test_behavior_returns_a_normalize_with_per_channel_means():
    images, loader = _data()
    norm = image_normalizer(loader)
    assert isinstance(norm, Normalize)
    expected = images.reshape(40, 3, -1).mean(2).mean(0)
    assert torch.allclose(torch.as_tensor(norm.mean).float(), expected, atol=1e-5)


def test_convention_std_is_the_average_of_per_image_stds():
    images, loader = _data()
    expected = images.reshape(40, 3, -1).std(2).mean(0)
    assert torch.allclose(torch.as_tensor(image_normalizer(loader).std).float(), expected, atol=1e-5)


def test_convention_differs_from_the_pooled_pixel_std():
    images, loader = _data()
    pooled = images.permute(1, 0, 2, 3).reshape(3, -1).std(1)
    assert not torch.allclose(torch.as_tensor(image_normalizer(loader).std).float(), pooled, atol=1e-3)
