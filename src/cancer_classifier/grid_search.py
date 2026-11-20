"""Grid search over learning rate, batch size and weight decay, choosing by validation accuracy.

Example:
    python -m cancer_classifier.grid_search --data data/lung_colon_image_set --epochs 3
"""

import argparse
import csv
import itertools
from pathlib import Path

from .data import make_loaders
from .engine import fit
from .model import CancerCNN
from .utils import get_device, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hyperparameter grid search.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/grid_search.csv"))
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64])
    p.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-4])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    device = get_device()
    rows = []
    grid = list(itertools.product(args.lrs, args.batch_sizes, args.weight_decays))
    for i, (lr, bs, wd) in enumerate(grid, 1):
        set_seed(args.seed)  # same split and starting weights for a fair comparison
        splits = make_loaders(args.data, bs, seed=args.seed, max_per_class=args.max_per_class)
        model = CancerCNN().to(device)
        print(f"\n[{i}/{len(grid)}] lr={lr} batch={bs} weight_decay={wd}")
        h = fit(model, splits, device, epochs=args.epochs, lr=lr, weight_decay=wd, patience=args.epochs)
        rows.append({"lr": lr, "batch_size": bs, "weight_decay": wd,
                     "best_val_acc": round(h["best_val_acc"], 4),
                     "best_val_loss": round(h["best_val_loss"], 4),
                     "seconds": h["train_seconds"]})

    rows.sort(key=lambda r: r["best_val_acc"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    print(f"\nBest: lr={best['lr']} batch={best['batch_size']} weight_decay={best['weight_decay']} "
          f"-> val acc {best['best_val_acc']:.2%}  (saved {args.out})")
    return rows


if __name__ == "__main__":
    main()
