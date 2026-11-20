"""Tests run on a tiny synthetic dataset, so they need no download and finish in seconds.

Each fake class gets a different dominant colour, so a working pipeline should learn it.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from cancer_classifier import CLASS_NAMES
from cancer_classifier import grid_search, predict, train
from cancer_classifier.data import find_images, make_loaders
from cancer_classifier.model import CancerCNN

PER_CLASS = 20
COLOURS = [(220, 40, 40), (40, 220, 40), (40, 40, 220), (220, 220, 40), (220, 40, 220)]


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("lc25000")
    rng = np.random.default_rng(0)
    # Mirror the Kaggle layout: lung_image_sets/lung_aca/... and colon_image_sets/colon_n/...
    for name, colour in zip(CLASS_NAMES, COLOURS):
        folder = root / f"{name.split('_')[0]}_image_sets" / name
        folder.mkdir(parents=True)
        for i in range(PER_CLASS):
            noise = rng.integers(-30, 30, size=(64, 64, 3))
            pixels = np.clip(np.array(colour) + noise, 0, 255).astype("uint8")
            Image.fromarray(pixels).save(folder / f"{name}{i}.jpeg")
    return root


def test_model_output_shape():
    model = CancerCNN()
    out = model(torch.randn(4, 3, 128, 128))
    assert out.shape == (4, len(CLASS_NAMES))


@pytest.mark.parametrize("size", [64, 128, 224])
def test_model_handles_other_image_sizes(size):
    assert CancerCNN(image_size=size)(torch.randn(2, 3, size, size)).shape == (2, 5)


def test_dropout_is_off_by_default():
    model = CancerCNN()
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)) for m in model.modules())
    assert any(isinstance(m, torch.nn.Dropout) for m in CancerCNN(dropout=0.5).modules())


def test_finds_images_in_kaggle_layout(fake_data):
    samples = find_images(fake_data)
    assert len(samples) == PER_CLASS * len(CLASS_NAMES)
    assert {label for _, label in samples} == set(range(len(CLASS_NAMES)))


def test_missing_dataset_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        find_images(tmp_path)


def test_splits_are_stratified_and_do_not_overlap(fake_data):
    splits = make_loaders(fake_data, batch_size=8, image_size=64)
    assert sum(splits.sizes.values()) == PER_CLASS * len(CLASS_NAMES)
    sets = [{str(p) for p, _ in loader.dataset.samples}
            for loader in (splits.train, splits.val, splits.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    test_labels = [label for _, label in splits.test.dataset.samples]
    assert set(test_labels) == set(range(len(CLASS_NAMES)))  # every class in the test set


def test_same_seed_gives_same_split(fake_data):
    a = make_loaders(fake_data, seed=7).test.dataset.samples
    b = make_loaders(fake_data, seed=7).test.dataset.samples
    assert a == b


def test_end_to_end_train_then_predict(fake_data, tmp_path):
    out = tmp_path / "run"
    results = train.main(["--data", str(fake_data), "--out", str(out), "--epochs", "4",
                          "--image-size", "64", "--batch-size", "8"])
    assert results["accuracy"] > 0.8  # colours are easy to tell apart
    for name in ("best_model.pt", "metrics.json", "training_curves.png", "confusion_matrix.png"):
        assert (out / name).exists()
    saved = json.loads((out / "metrics.json").read_text())
    assert set(saved["test"]["per_class"]) == set(CLASS_NAMES)

    image = next((fake_data / "lung_image_sets" / "lung_n").iterdir())
    pred = predict.main([str(image), "--model", str(out / "best_model.pt"), "--image-size", "64"])
    assert abs(sum(pred[0]["probabilities"].values()) - 1) < 1e-4


def test_grid_search_ranks_configs(fake_data, tmp_path):
    rows = grid_search.main(["--data", str(fake_data), "--out", str(tmp_path / "grid.csv"),
                             "--lrs", "1e-3", "1e-4", "--batch-sizes", "8",
                             "--weight-decays", "0", "--epochs", "1"])
    assert len(rows) == 2
    assert rows[0]["best_val_acc"] >= rows[1]["best_val_acc"]
    assert (tmp_path / "grid.csv").exists()
