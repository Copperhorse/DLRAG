import os, re
from common import make_model, make_loader
from solution import train_with_tensorboard


def _run(tmp_path, n=3):
    return train_with_tensorboard(make_model(), make_loader(shuffle=True), make_loader(seed=1), n, "myrun", str(tmp_path))


def test_behavior_a_tensorboard_event_file_was_written_in_the_requested_folder(tmp_path):
    out = _run(tmp_path)
    files = [f for _, _, fs in os.walk(tmp_path) for f in fs]
    assert out is not None and any(f.startswith("events.out.tfevents") for f in files)
    assert len(out.losses) == 3 and len(out.val_losses) == 3


def test_convention_run_directory_is_run_name_plus_timestamp_as_the_books_class_names_it(tmp_path):
    _run(tmp_path)
    dirs = [d for d in os.listdir(tmp_path) if os.path.isdir(tmp_path / d)]
    assert len(dirs) == 1 and re.fullmatch(r"myrun_\d{14}", dirs[0])


def test_convention_the_returned_object_is_the_books_training_class_with_a_writer(tmp_path):
    out = _run(tmp_path)
    assert type(out).__name__ == "StepByStep" and out.writer is not None and out.total_epochs == 3
