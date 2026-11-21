"""Grad-CAM: show which parts of a tissue image the model used to make its prediction.

How it works, in short:
1. Run an image through the model and keep the feature maps from the last conv layer
   (64 maps of 32x32 for a 128x128 image). Each map lights up where it found a pattern.
2. Backpropagate the score of the predicted class to those maps. The average gradient of
   each map says how much that map pushed the prediction up.
3. Weight each map by that importance, add them up, and keep only the positive part.
   The result is a coarse heatmap of the regions that supported the prediction.
4. Upscale the heatmap to the image size and draw it on top of the image.

Reference: Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", arXiv:1610.02391.

Example:
    python -m cancer_classifier.gradcam --model results/run/best_model.pt --data data \
        --out results/run/gradcam.png
"""

import argparse
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from . import CLASS_LABELS, CLASS_NAMES  # noqa: E402
from .data import build_transforms, make_loaders  # noqa: E402
from .model import CancerCNN  # noqa: E402
from .utils import get_device  # noqa: E402


class GradCAM:
    """Grad-CAM for CancerCNN, using the output of the last conv block's ReLU."""

    def __init__(self, model: CancerCNN):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        last_relu = model.features[-1][1]  # Conv2d -> ReLU -> MaxPool2d in the last block
        last_relu.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad

    def __call__(self, image: torch.Tensor, target_class: int | None = None):
        """Return (heatmap in [0, 1] at image size, predicted class, class probabilities).

        `image` is one normalised tensor of shape (3, H, W). By default the heatmap explains
        the predicted class; pass `target_class` to explain a different one.
        """
        x = image.unsqueeze(0).requires_grad_(True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach()
        predicted = int(probs.argmax())
        target = predicted if target_class is None else target_class

        self.model.zero_grad()
        logits[0, target].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)       # importance of each map
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)        # scale to [0, 1]
        return cam.cpu(), predicted, probs.cpu()


def to_display(image: torch.Tensor):
    """Undo the [-1, 1] normalisation so the image can be shown."""
    return (image * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def pick_test_images(data_dir: Path, image_size: int, per_class: int, mistakes: int,
                     model: CancerCNN, device, seed: int = 42):
    """Pick images from the held-out test split (same seed as training, so never seen).

    Returns `per_class` correctly classified images per class, plus up to `mistakes`
    misclassified ones, which are the most interesting to look at.
    """
    test_samples = make_loaders(data_dir, image_size=image_size, seed=seed).test.dataset.samples
    tf = build_transforms(image_size)
    correct = {c: [] for c in range(len(CLASS_NAMES))}
    wrong = []
    model.eval()
    with torch.no_grad():
        for path, label in test_samples:
            if all(len(v) >= per_class for v in correct.values()) and len(wrong) >= mistakes:
                break
            tensor = tf(Image.open(path).convert("RGB"))
            pred = int(model(tensor.unsqueeze(0).to(device)).argmax())
            if pred == label and len(correct[label]) < per_class:
                correct[label].append((path, label))
            elif pred != label and len(wrong) < mistakes:
                wrong.append((path, label))
    picked = [s for c in sorted(correct) for s in correct[c]]
    return picked + wrong


def plot_gradcam(cam_tool: GradCAM, samples, image_size: int, device, path: Path) -> None:
    """One row per image: the original tissue, then the heatmap laid over it."""
    tf = build_transforms(image_size)
    fig, axes = plt.subplots(len(samples), 2, figsize=(6, 3 * len(samples)), squeeze=False)
    for row, (img_path, label) in zip(axes, samples):
        image = tf(Image.open(img_path).convert("RGB")).to(device)
        cam, pred, probs = cam_tool(image)
        shown = to_display(image)
        correct = label is None or pred == label
        truth = f"True: {CLASS_NAMES[label]}\n" if label is not None else ""
        row[0].imshow(shown)
        row[0].set_title(f"{truth}Pred: {CLASS_NAMES[pred]} ({probs[pred]:.0%})",
                         fontsize=9, color="black" if correct else "red")
        row[1].imshow(shown)
        row[1].imshow(cam.numpy(), cmap="jet", alpha=0.45)
        row[1].set_title("Grad-CAM", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description="Draw Grad-CAM heatmaps for a trained model.")
    p.add_argument("images", type=Path, nargs="*",
                   help="images to explain (if none, picks from the test split of --data)")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, help="dataset folder, used when no images are given")
    p.add_argument("--per-class", type=int, default=1, help="correct examples per class")
    p.add_argument("--mistakes", type=int, default=3, help="misclassified examples to add")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("results/gradcam.png"))
    args = p.parse_args(argv)

    device = get_device()
    model = CancerCNN(image_size=args.image_size).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    if args.images:
        samples = [(path, None) for path in args.images]
    elif args.data:
        samples = pick_test_images(args.data, args.image_size, args.per_class, args.mistakes,
                                   model, device)
    else:
        p.error("give some image paths or --data")

    plot_gradcam(GradCAM(model), samples, args.image_size, device, args.out)
    print(f"Saved Grad-CAM figure for {len(samples)} images to {args.out}")
    for path, label in samples:
        if label is not None:
            print(f"  {Path(path).name}: true {CLASS_LABELS[CLASS_NAMES[label]]}")
    return args.out


if __name__ == "__main__":
    main()
