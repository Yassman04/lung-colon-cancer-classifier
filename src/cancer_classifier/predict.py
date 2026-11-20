"""Classify one or more images with a trained model.

Example:
    python -m cancer_classifier.predict --model results/run/best_model.pt path/to/image.jpeg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from . import CLASS_LABELS, CLASS_NAMES
from .data import build_transforms
from .model import CancerCNN
from .utils import get_device


@torch.no_grad()
def classify(model, image_path: Path, image_size: int, device) -> dict[str, float]:
    tensor = build_transforms(image_size)(Image.open(image_path).convert("RGB")).unsqueeze(0)
    probs = torch.softmax(model(tensor.to(device)), dim=1)[0].cpu()
    return {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description="Classify histopathology images.")
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=128)
    args = p.parse_args(argv)

    device = get_device()
    model = CancerCNN(image_size=args.image_size).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    outputs = []
    for path in args.images:
        probs = classify(model, path, args.image_size, device)
        top = max(probs, key=probs.get)
        print(f"{path.name}: {CLASS_LABELS[top]} ({probs[top]:.1%} confidence)")
        outputs.append({"image": str(path), "prediction": top, "probabilities": probs})
    return outputs


if __name__ == "__main__":
    main()
