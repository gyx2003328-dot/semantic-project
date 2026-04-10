#!/usr/bin/env python3
"""
Evaluate a trained UNet checkpoint on a test split and export leaderboard JSON.

Metrics:
- Dice Score (mean Dice over valid classes)
- mIoU (mean IoU over valid classes)
- FWIoU (frequency weighted IoU)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return torch.device(device_arg)
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SplitDataset(Dataset):
    """Dataset from explicit split file with fixed mask mapping."""

    def __init__(self, imgs_dir: Path, masks_dir: Path, split_file: Path, mask_values: list, scale: float):
        self.imgs_dir = Path(imgs_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.mask_values = mask_values
        self.ids = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        img_path = self.imgs_dir / f"{sample_id}.jpg"
        if not img_path.exists():
            img_path = self.imgs_dir / f"{sample_id}.png"
        mask_path = self.masks_dir / f"{sample_id}.png"

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        assert img.size == mask.size, f"Image/mask size mismatch: {sample_id}"

        img_arr = BasicDataset.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask_arr = BasicDataset.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            "image": torch.as_tensor(img_arr.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask_arr.copy()).long().contiguous(),
        }


def build_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


def calculate_all_metrics(confusion_matrix: np.ndarray) -> dict:
    hist = confusion_matrix
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = intersection / (union + 1e-10)
    dice = 2 * intersection / (hist.sum(axis=1) + hist.sum(axis=0) + 1e-10)
    freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
    valid = hist.sum(axis=1) > 0

    miou = np.mean(iou[valid]) if np.any(valid) else 0.0
    mdice = np.mean(dice[valid]) if np.any(valid) else 0.0
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()

    pixel_acc = intersection.sum() / (hist.sum() + 1e-10)
    class_acc = intersection / (hist.sum(axis=1) + 1e-10)
    mean_acc = np.mean(class_acc[valid]) if np.any(valid) else 0.0

    return {
        "dice_score": round(float(mdice) * 100, 2),
        "miou": round(float(miou) * 100, 2),
        "fwiou": round(float(fwiou) * 100, 2),
        "pixel_accuracy": round(float(pixel_acc) * 100, 2),
        "mean_accuracy": round(float(mean_acc) * 100, 2),
        "iou_per_class": [round(float(x) * 100, 2) for x in iou.tolist()],
        "dice_per_class": [round(float(x) * 100, 2) for x in dice.tolist()],
        "freq_per_class": [round(float(x) * 100, 2) for x in freq.tolist()],
    }


DEFAULT_CLASS_NAME_MAP = {
    0: "ground",
    1: "roof",
    2: "building",
    4: "river",
    11: "road",
    13: "green_field",
    14: "wild_field",
    20: "sedan",
}


def parse_epoch_from_checkpoint(path: str) -> int:
    m = re.search(r"epoch(\d+)", Path(path).stem)
    return int(m.group(1)) if m else 0


@torch.inference_mode()
def evaluate_checkpoint(
    model: UNet,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        image = batch["image"].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        target = batch["mask"].numpy()

        pred_logits = model(image)
        pred = pred_logits.argmax(dim=1).cpu().numpy()

        for p, t in zip(pred, target):
            confusion += build_confusion_matrix(p, t, num_classes)

    return calculate_all_metrics(confusion)


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNet and export leaderboard JSON")
    parser.add_argument("--model", required=True, type=str, help="Path to checkpoint .pth")
    parser.add_argument("--imgs-dir", default="data/imgs", type=str, help="Images directory")
    parser.add_argument("--masks-dir", default="data/masks", type=str, help="Masks directory")
    parser.add_argument("--split-file", default="data/uavscenes_test.txt", type=str, help="Test split txt")
    parser.add_argument("--scale", default=0.25, type=float, help="Image scale")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for evaluation")
    parser.add_argument("--output", default="output/training_report.json", type=str, help="Output JSON path")
    parser.add_argument("--team", default="Team Alpha", type=str, help="Team name")
    parser.add_argument("--repo-url", default="https://github.com/xxxxxx.git", type=str, help="Private repo URL")
    parser.add_argument("--device", default="auto", type=str, help="auto|cuda|cpu")
    parser.add_argument("--history-file", default="output/train_history.json", type=str, help="Training history JSON from train.py")
    parser.add_argument("--train-split-file", default="data/uavscenes_train.txt", type=str, help="Train split txt")
    parser.add_argument("--val-split-file", default="data/uavscenes_val.txt", type=str, help="Val split txt")
    parser.add_argument("--train-batch-size", default=2, type=int, help="Fallback train batch size when history is missing")
    parser.add_argument("--train-learning-rate", default=1e-4, type=float, help="Fallback train learning rate when history is missing")
    parser.add_argument("--train-optimizer", default="RMSprop", type=str, help="Fallback optimizer when history is missing")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    checkpoint = torch.load(model_path, map_location=device)
    mask_values = checkpoint.get("mask_values")
    if mask_values is None:
        raise RuntimeError("Checkpoint missing 'mask_values'. Please use checkpoints saved by train.py")

    n_classes = len(mask_values)
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=False)
    model.load_state_dict({k: v for k, v in checkpoint.items() if k != "mask_values"})
    model.to(device=device, memory_format=torch.channels_last)

    dataset = SplitDataset(
        imgs_dir=Path(args.imgs_dir),
        masks_dir=Path(args.masks_dir),
        split_file=Path(args.split_file),
        mask_values=mask_values,
        scale=args.scale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    metrics = evaluate_checkpoint(model, loader, device, n_classes)

    history = {}
    history_path = Path(args.history_file)
    if history_path.exists():
        history = json.loads(history_path.read_text())

    train_count = 0
    val_count = 0
    train_split = Path(args.train_split_file)
    val_split = Path(args.val_split_file)
    if train_split.exists():
        train_count = len([x for x in train_split.read_text().splitlines() if x.strip()])
    if val_split.exists():
        val_count = len([x for x in val_split.read_text().splitlines() if x.strip()])
    test_count = len(dataset)

    mask_values = [int(x) for x in mask_values]
    class_mapping = {str(v): DEFAULT_CLASS_NAME_MAP.get(v, f"class_{v}") for v in mask_values}

    iou_list = metrics["iou_per_class"]
    dice_list = metrics["dice_per_class"]
    freq_list = metrics["freq_per_class"]
    per_class_results = {}
    for idx, orig_id in enumerate(mask_values):
        class_name = class_mapping[str(orig_id)]
        per_class_results[class_name] = {
            "iou": round(float(iou_list[idx]), 2),
            "dice": round(float(dice_list[idx]), 2),
            "frequency": round(float(freq_list[idx]), 2),
        }

    epoch_values = history.get("epochs", [])
    loss_values = history.get("train_loss", [])
    val_dice_values = [v for v in history.get("val_dice", []) if v is not None]
    total_epochs = int(epoch_values[-1]) if epoch_values else parse_epoch_from_checkpoint(args.model)
    final_train_loss = float(loss_values[-1]) if loss_values else 0.0
    final_val_dice = float(val_dice_values[-1]) if val_dice_values else 0.0

    report = {
        "training_summary": {
            "total_epochs": total_epochs,
            "total_images": train_count + val_count + test_count,
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "num_classes": n_classes,
            "image_scale": float(args.scale),
            "batch_size": int(history.get("meta", {}).get("batch_size", args.train_batch_size)),
            "learning_rate": float(history.get("meta", {}).get("learning_rate", args.train_learning_rate)),
            "optimizer": str(history.get("meta", {}).get("optimizer", args.train_optimizer)),
            "final_train_loss": round(final_train_loss, 4),
            "final_val_dice": round(final_val_dice, 4),
        },
        "test_metrics": {
            "dice_score": metrics["dice_score"],
            "miou": metrics["miou"],
            "fwiou": metrics["fwiou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "mean_accuracy": metrics["mean_accuracy"],
        },
        "per_class_results": per_class_results,
        "class_mapping": class_mapping,
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
