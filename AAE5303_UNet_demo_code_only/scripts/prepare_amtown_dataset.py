#!/usr/bin/env python3
"""Prepare AMtown dataset layout:
- Train set: AMtown01 + AMtown03
- Test set: AMtown02
- Validation split: 15% from train pool
"""

import argparse
import random
import zipfile
from pathlib import Path


def ensure_labels_extracted(labels_zip: Path, labels_root: Path):
    labels_root.mkdir(parents=True, exist_ok=True)
    marker = labels_root / ".extract_done"
    if marker.exists():
        return

    wanted_prefixes = [
        "interval5_CAM_label/interval5_AMtown01/interval5_CAM_label_id/",
        "interval5_CAM_label/interval5_AMtown02/interval5_CAM_label_id/",
        "interval5_CAM_label/interval5_AMtown03/interval5_CAM_label_id/",
    ]
    with zipfile.ZipFile(labels_zip) as zf:
        members = [n for n in zf.namelist() if any(n.startswith(p) for p in wanted_prefixes)]
        zf.extractall(labels_root, members=members)
    marker.write_text("ok\n")


def collect_stem_to_path(img_dir: Path, mask_dir: Path):
    imgs = {p.stem: p for p in img_dir.glob("*.jpg")}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}
    common = sorted(set(imgs.keys()) & set(masks.keys()))
    return common, imgs, masks


def rebuild_links(out_img_dir: Path, out_mask_dir: Path, stems, imgs_map, masks_map):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # clear old links/files
    for p in out_img_dir.glob("*"):
        if p.is_symlink() or p.is_file():
            p.unlink()
    for p in out_mask_dir.glob("*"):
        if p.is_symlink() or p.is_file():
            p.unlink()

    for s in stems:
        (out_img_dir / f"{s}.jpg").symlink_to(imgs_map[s])
        (out_mask_dir / f"{s}.png").symlink_to(masks_map[s])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/data/AAE5303_UNet_demo")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.project_root)
    labels_zip = root / "data_downloads" / "AMtown01_labels"
    labels_root = root / "data_raw" / "AMtown_labels"
    ensure_labels_extracted(labels_zip, labels_root)

    am1_img = root / "data_raw" / "AMtown01_images" / "interval5_AMtown01" / "interval5_CAM"
    am2_img = root / "data_raw" / "AMtown02_images" / "interval5_AMtown02" / "interval5_CAM"
    am3_img = root / "data_raw" / "AMtown03_images" / "interval5_AMtown03" / "interval5_CAM"

    am1_mask = labels_root / "interval5_CAM_label" / "interval5_AMtown01" / "interval5_CAM_label_id"
    am2_mask = labels_root / "interval5_CAM_label" / "interval5_AMtown02" / "interval5_CAM_label_id"
    am3_mask = labels_root / "interval5_CAM_label" / "interval5_AMtown03" / "interval5_CAM_label_id"

    s1, i1, m1 = collect_stem_to_path(am1_img, am1_mask)
    s2, i2, m2 = collect_stem_to_path(am2_img, am2_mask)
    s3, i3, m3 = collect_stem_to_path(am3_img, am3_mask)

    train_stems = sorted(s1 + s3)
    test_stems = sorted(s2)
    train_imgs = {**i1, **i3}
    train_masks = {**m1, **m3}

    data_dir = root / "data"
    train_imgs_dir = data_dir / "amtown_train_imgs"
    train_masks_dir = data_dir / "amtown_train_masks"
    test_imgs_dir = data_dir / "amtown_test_imgs"
    test_masks_dir = data_dir / "amtown_test_masks"

    rebuild_links(train_imgs_dir, train_masks_dir, train_stems, train_imgs, train_masks)
    rebuild_links(test_imgs_dir, test_masks_dir, test_stems, i2, m2)

    # 85/15 split for training pool
    random.seed(args.seed)
    shuffled = train_stems[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * args.val_ratio)
    val_split = sorted(shuffled[:n_val])
    train_split = sorted(shuffled[n_val:])

    (data_dir / "amtown_train.txt").write_text("\n".join(train_split) + "\n")
    (data_dir / "amtown_val.txt").write_text("\n".join(val_split) + "\n")
    (data_dir / "amtown_test.txt").write_text("\n".join(test_stems) + "\n")

    print(f"AMtown01 matched: {len(s1)}")
    print(f"AMtown03 matched: {len(s3)}")
    print(f"Train pool (01+03): {len(train_stems)}")
    print(f"Split -> train: {len(train_split)}, val: {len(val_split)}")
    print(f"Test (AMtown02): {len(test_stems)}")
    print(f"train imgs dir: {train_imgs_dir}")
    print(f"test imgs dir: {test_imgs_dir}")


if __name__ == "__main__":
    main()

