"""Train the CNN and evaluate it once on the held-out test set.

Example:
    python -m cancer_classifier.train --data data/lung_colon_image_set --epochs 10
"""

import argparse
from pathlib import Path

import torch

from .data import make_loaders
from .engine import fit, metrics, predict
from .model import CancerCNN, count_parameters
from .plots import plot_confusion, plot_history
from .utils import get_device, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the lung/colon cancer CNN.")
    p.add_argument("--data", type=Path, required=True, help="folder containing the 5 class folders")
    p.add_argument("--out", type=Path, default=Path("results/run"), help="where to save outputs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0, help="0 = off (best in testing)")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--augment", action="store_true", help="random flips/rotations on training images")
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    p.add_argument("--max-per-class", type=int, default=None, help="use fewer images for a quick run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    args.out.mkdir(parents=True, exist_ok=True)

    splits = make_loaders(args.data, args.batch_size, args.image_size, args.seed,
                          augment=args.augment, max_per_class=args.max_per_class,
                          num_workers=args.workers)
    model = CancerCNN(image_size=args.image_size, dropout=args.dropout).to(device)
    print(f"Device: {device} | splits: {splits.sizes} | parameters: {count_parameters(model):,}")

    history = fit(model, splits, device, epochs=args.epochs, lr=args.lr,
                  weight_decay=args.weight_decay, patience=args.patience,
                  checkpoint=args.out / "best_model.pt")

    y_true, y_pred = predict(model, splits.test, device)
    results = metrics(y_true, y_pred)
    print(f"\nTEST  accuracy {results['accuracy']:.2%} | macro F1 {results['macro_f1']:.2%}")

    plot_history(history, args.out / "training_curves.png")
    plot_confusion(results["confusion_matrix"], args.out / "confusion_matrix.png")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    save_json({"config": config, "splits": splits.sizes, "history": history, "test": results},
              args.out / "metrics.json")
    torch.save(model.state_dict(), args.out / "best_model.pt")
    print(f"Saved model, metrics and plots to {args.out}/")
    return results


if __name__ == "__main__":
    main()
