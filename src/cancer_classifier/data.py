"""Loading the LC25000 dataset and splitting it into train / validation / test sets."""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Inputs are scaled to roughly [-1, 1], which keeps training stable with Adam.
NORMALISE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def build_transforms(image_size: int = 128, augment: bool = False) -> transforms.Compose:
    """Resize + normalise. Optional flips/rotations for training only.

    Tissue has no 'right way up', so flips and 90-degree rotations are safe augmentations.
    """
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        ]
    steps += [transforms.ToTensor(), NORMALISE]
    return transforms.Compose(steps)


def find_images(root: Path, max_per_class: int | None = None) -> list[tuple[Path, int]]:
    """Return (image_path, label) pairs from a folder with one sub-folder per class.

    Accepts either the folder that directly holds the five class folders, or the
    Kaggle download layout (lung_image_sets/ and colon_image_sets/).
    """
    root = Path(root)
    samples: list[tuple[Path, int]] = []
    for label, class_name in enumerate(CLASS_NAMES):
        matches = sorted(p for p in root.rglob(class_name) if p.is_dir())
        if not matches:
            raise FileNotFoundError(
                f"Could not find a '{class_name}' folder under {root}. "
                "See the README for how to download the dataset."
            )
        files = sorted(
            f for f in matches[0].iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_per_class:
            files = files[:max_per_class]
        samples.extend((f, label) for f in files)
    return samples


class HistopathologyDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


@dataclass
class DataSplits:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    sizes: dict


def make_loaders(
    data_dir: Path,
    batch_size: int = 32,
    image_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    augment: bool = False,
    max_per_class: int | None = None,
    num_workers: int = 0,
) -> DataSplits:
    """Stratified 70/15/15 split, so every split has the same mix of the five classes.

    The test set is only used once, at the very end; all tuning uses the validation set.
    """
    samples = find_images(data_dir, max_per_class)
    labels = [label for _, label in samples]

    train_val, test = train_test_split(
        samples, test_size=test_frac, stratify=labels, random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_frac / (1 - test_frac),
        stratify=[label for _, label in train_val],
        random_state=seed,
    )

    eval_tf = build_transforms(image_size, augment=False)
    train_tf = build_transforms(image_size, augment=augment)

    g = torch.Generator().manual_seed(seed)
    loader = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=g
    )
    return DataSplits(
        train=loader(HistopathologyDataset(train, train_tf), True),
        val=loader(HistopathologyDataset(val, eval_tf), False),
        test=loader(HistopathologyDataset(test, eval_tf), False),
        sizes={"train": len(train), "val": len(val), "test": len(test)},
    )
