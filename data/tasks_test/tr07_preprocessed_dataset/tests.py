import torch, torch.nn as nn
from torch.utils.data import TensorDataset
from common import make_loader, make_model
from solution import preprocessed_dataset


def test_behavior_features_and_labels_cover_all_samples_in_order():
    model = make_model(); loader = make_loader(batch_size=16, n=40)
    ds = preprocessed_dataset(model, loader)
    xs = torch.cat([x for x, _ in loader]); ys = torch.cat([y for _, y in loader])
    assert isinstance(ds, TensorDataset) and len(ds) == 40
    with torch.no_grad():
        assert torch.allclose(ds.tensors[0], model(xs), atol=1e-6)
    assert torch.equal(ds.tensors[1], ys)


def test_convention_outputs_are_detached_cpu_tensors():
    ds = preprocessed_dataset(make_model(), make_loader())
    assert all(not t.requires_grad and t.device.type == "cpu" for t in ds.tensors)


def test_convention_model_is_left_in_evaluation_mode():
    model = make_model(); model.train()
    preprocessed_dataset(model, make_loader())
    assert not model.training
