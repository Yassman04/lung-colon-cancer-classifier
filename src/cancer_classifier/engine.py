"""Training and evaluation loops, kept separate from the command-line scripts so they can be tested."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import CLASS_NAMES


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass over the data. Trains if an optimiser is given, otherwise just evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def fit(model, splits, device, epochs=10, lr=1e-3, weight_decay=0.0, patience=3,
        checkpoint: Path | None = None, log=print) -> dict:
    """Train with Adam, keep the best model by validation loss, stop early if it stops improving."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, splits.train, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, splits.val, criterion, device)
        for key, value in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[key].append(value)
        log(f"Epoch {epoch:>2}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.2%} "
            f"| val loss {va_loss:.4f} acc {va_acc:.2%}")

        if va_loss < best_loss:
            best_loss, bad_epochs = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"Early stopping: no val improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_loss
    history["best_val_acc"] = max(history["val_acc"])
    history["epochs_run"] = len(history["val_acc"])
    history["train_seconds"] = round(time.time() - start, 1)
    return history


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_true += labels.tolist()
        y_pred += logits.argmax(1).cpu().tolist()
    return y_true, y_pred


def metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro-averaged scores (each class counts equally) and per-class results."""
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASS_NAMES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
