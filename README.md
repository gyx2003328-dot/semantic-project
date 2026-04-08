# semantic-project
# AAE5303 Assignment 3: Semantic Segmentation with UNet (Run Report)

**2D semantic segmentation using UNet on UAV aerial imagery**

**This project uses Dataset 1—the original release as provided by the teaching assistants. Because the raw files are very large, they are not re-uploaded to this repository. Please download Dataset 1 from the official course materials (or as instructed by the TA), then put the images and masks in the expected folders (e.g. data/imgs/ and data/masks/) before training or evaluation.

_UAVScenes HKisland dataset_

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Reproducible Commands](#reproducible-commands)
5. [Training Configuration](#training-configuration)
6. [Training Process](#training-process)
7. [Evaluation Results (Test Set)](#evaluation-results-test-set)
8. [Per-Class Results](#per-class-results)
9. [Generated Visualizations](#generated-visualizations)
10. [Files Produced](#files-produced)

---

## Executive Summary

This document is an English run report for a UNet semantic segmentation project on the **UAVScenes HKisland** dataset. It is written to match the **actual execution logs and outputs** from this workspace (`train_and_eval.log`, `output/train_history.json`, and `output/training_report.json`).

### Key Results (Test Set)

| Item | Value |
|------|-------|
| **Epochs** | 10 |
| **Classes** | 8 |
| **Scale** | 0.35 (3840×2160 → 1344×756) |
| **Device** | CUDA |
| **Dice** | 88.57% |
| **mIoU** | 80.24% |
| **FWIoU** | 77.44% |
| **Pixel Accuracy** | 86.46% |
| **Mean Accuracy** | 92.25% |
| **Best Val Dice (epoch)** | 0.7730 (epoch 6) |

---

## Project Overview

The goal is pixel-wise semantic segmentation using a UNet encoder–decoder model with skip connections. The workflow is:

- Prepare `data/imgs` and `data/masks`
- Train UNet with mixed precision (AMP) on GPU
- Track train loss and validation Dice across epochs
- Evaluate on the test split and export a JSON report
- Generate plots and a dashboard summarizing results

---

## Dataset

### Dataset Summary

| Property | Value |
|----------|-------|
| **Dataset Name** | UAVScenes HKisland |
| **Total Samples** | 1,356 |
| **Train / Val / Test** | 1,153 / 203 / 204 |
| **Num Classes** | 8 |
| **Unique Mask Values** | `[0, 1, 2, 4, 11, 13, 14, 20]` |

### Class Mapping (original → name)

| Original ID | Class Name |
|-------------|------------|
| 0 | ground |
| 1 | roof |
| 2 | building |
| 4 | river |
| 11 | road |
| 13 | green_field |
| 14 | wild_field |
| 20 | sedan |

---

## Reproducible Commands

### Training

```bash
python train.py --epochs 10 --batch-size 2 --learning-rate 0.0001 --scale 0.35 --validation 15 --classes 8 --device auto --amp --stable-mode --num-workers -1 --optimizer adamw --scheduler cosine --loss-name ce_dice --weighted-loss --augmentation --weighted-sampler
```

### Evaluation / Submission Report

```bash
python evaluate_submission.py --model checkpoints/checkpoint_epoch10.pth --imgs-dir data/imgs --masks-dir data/masks --split-file data/uavscenes_test.txt --scale 0.35 --output output/training_report.json --team YourTeam --repo-url https://github.com/xxxxxx.git --device auto
```

---

## Training Configuration

| Setting | Value |
|--------|-------|
| **Epochs** | 10 |
| **Batch size** | 2 |
| **Learning rate** | 1e-4 |
| **Optimizer** | AdamW |
| **Scheduler** | Cosine |
| **Loss** | `ce_dice` (Cross-Entropy + Dice) |
| **AMP** | Enabled |
| **Stable mode** | Enabled |
| **Augmentation** | Enabled |
| **Weighted loss** | Enabled |
| **Weighted sampler** | Enabled |
| **Scale** | 0.35 |
| **Device** | CUDA (auto-selected) |
| **DataLoader workers** | 4 (resolved from `--num-workers -1`) |
| **pin_memory** | True |

---

## Training Process

Training history is saved to `output/train_history.json`. The best validation Dice occurred at **epoch 6**.

| Epoch | Train Loss | Val Dice |
|------:|-----------:|---------:|
| 1 | 0.9417 | 0.6475 |
| 2 | 0.3505 | 0.6817 |
| 3 | 0.2449 | 0.6936 |
| 4 | 0.1925 | 0.5885 |
| 5 | 0.1682 | 0.7649 |
| 6 | 0.1319 | **0.7730** |
| 7 | 0.1107 | 0.7341 |
| 8 | 0.1071 | 0.7507 |
| 9 | 0.0991 | 0.7364 |
| 10 | 0.0948 | 0.7460 |

---

## Evaluation Results (Test Set)

Results are recorded in `output/training_report.json`.

| Metric | Value |
|--------|-------|
| **Dice** | 88.57% |
| **mIoU** | 80.24% |
| **FWIoU** | 77.44% |
| **Pixel Accuracy** | 86.46% |
| **Mean Accuracy** | 92.25% |

---

## Per-Class Results

| Class | IoU | Dice | Frequency |
|-------|-----|------|-----------|
| ground | 55.23% | 71.16% | 20.22% |
| roof | 87.90% | 93.56% | 1.83% |
| building | 87.85% | 93.53% | 0.37% |
| river | 78.39% | 87.89% | 31.07% |
| road | 84.83% | 91.79% | 0.41% |
| green_field | 93.79% | 96.79% | 18.10% |
| wild_field | 80.95% | 89.47% | 27.94% |
| sedan | 72.99% | 84.38% | 0.05% |

---

## Generated Visualizations

The following plots are generated and saved under `figures/`:

- `figures/training_loss_curve.png`
- `figures/class_distribution.png`
- `figures/per_class_iou.png`
- `figures/summary_dashboard.png`

---

## Files Produced

Key output artifacts from the run:

- `train_and_eval.log`
- `output/train_history.json`
- `output/training_report.json`
- `checkpoints/checkpoint_epoch10.pth`
- `figures/*.png`
- `docs/PROJECT_REPORT.md`


