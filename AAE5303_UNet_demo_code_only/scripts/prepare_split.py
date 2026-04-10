#!/usr/bin/env python3
"""Prepare train/val/test split files from matched image/mask stems."""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-dir", required=True, type=str)
    parser.add_argument("--masks-dir", required=True, type=str)
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out-dir", default="data", type=str)
    args = parser.parse_args()

    imgs_dir = Path(args.imgs_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_stems = {p.stem for p in imgs_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    mask_stems = {p.stem for p in masks_dir.glob("*.png")}
    stems = sorted(img_stems & mask_stems)

    random.seed(args.seed)
    random.shuffle(stems)

    n = len(stems)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = sorted(stems[:n_train])
    val = sorted(stems[n_train:n_train + n_val])
    test = sorted(stems[n_train + n_val:])

    (out_dir / "uavscenes_train.txt").write_text("\n".join(train) + "\n")
    (out_dir / "uavscenes_val.txt").write_text("\n".join(val) + "\n")
    (out_dir / "uavscenes_test.txt").write_text("\n".join(test) + "\n")

    print(f"matched={n}, train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
