"""Small helpers shared by the training, evaluation and prediction scripts."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make runs repeatable: same seed -> same split, same weights, same results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use a GPU if one is available (NVIDIA or Apple Silicon), otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
