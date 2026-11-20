"""Saving training curves and the confusion matrix as images."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from . import CLASS_NAMES  # noqa: E402


def plot_history(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(title="Loss", xlabel="epoch")
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="validation")
    ax2.set(title="Accuracy (%)", xlabel="epoch")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], path: Path, title: str = "Test set") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, values_format="d", colorbar=False)
    ax.set_title(f"{title} confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
