# semantic-project
# AAE5303 UNet Segmentation Project 

This repository contains a **reproducible UNet training + evaluation pipeline** for **semantic segmentation** and produces:

- Model checkpoints in `checkpoints/`
- A standardized evaluation report in `output/training_report.json`
- Training history in `output/train_history.json`
- Plots in `figures/`
- A generated project report in `docs/PROJECT_REPORT.md`

This README is written to match the **actual execution artifacts** in this workspace (especially `output/training_report.json` and `output/train_history.json`). For an execution-oriented run log, see `README_RUN_EN.md`.

## Key results (Test set)

From `output/training_report.json`:

| Metric | Value |
|---|---:|
| **Dice** | **88.79%** |
| **mIoU** | **81.15%** |
| **FWIoU** | **92.15%** |
| **Pixel Accuracy** | **95.83%** |
| **Mean Accuracy** | **96.22%** |

## Project overview

Goal: pixel-wise semantic segmentation using a **UNet encoder–decoder** with skip connections.

End-to-end workflow:

- Prepare dataset folders and split files
- Train UNet (optionally with AMP, stable mode, and imbalance-handling strategies)
- Track training loss and validation Dice by epoch
- Evaluate on a test split and export a standard JSON report
- Generate plots and a short markdown report

## Dataset and splits

This project expects paired images and masks with aligned filenames, and uses explicit split `.txt` files (one sample id per line).

The default one-command pipeline uses AMtown-style directories and split files created by `scripts/prepare_amtown_dataset.py`:

- Training:
  - `data/amtown_train_imgs/`
  - `data/amtown_train_masks/`
  - `data/amtown_train.txt`
  - `data/amtown_val.txt`
- Test:
  - `data/amtown_test_imgs/`
  - `data/amtown_test_masks/`
  - `data/amtown_test.txt`

## Reproducible commands

### Option A: one-command pipeline (recommended)

Run the full pipeline (prepare data → train → eval → plots → report):

```bash
bash scripts/run_pipeline.sh
```

Notes:

- On Windows, run via **WSL2** or **Git Bash** (the pipeline is a Bash script).
- The pipeline uses `--device auto` (GPU if available, otherwise CPU).

### Option B: run stages separately

Train:

```bash
bash scripts/run_pipeline.sh train --epochs 30 --batch-size 4 --scale 0.35
```

Evaluate:

```bash
bash scripts/run_pipeline.sh eval --model checkpoints/best_checkpoint.pth
```

Generate plots:

```bash
bash scripts/run_pipeline.sh visualize
```

Generate the markdown report:

```bash
bash scripts/run_pipeline.sh report
```

### Option C: run Python entrypoints directly (no Bash)

Train:

```bash
python train.py \
  --img-dir data/amtown_train_imgs \
  --mask-dir data/amtown_train_masks \
  --epochs 30 \
  --batch-size 4 \
  --learning-rate 0.0001 \
  --scale 0.35 \
  --validation 15 \
  --classes 15 \
  --device auto \
  --amp \
  --stable-mode \
  --num-workers -1 \
  --optimizer adamw \
  --scheduler cosine \
  --loss-name ce_dice \
  --weighted-loss \
  --augmentation \
  --weighted-sampler \
  --history-out output/train_history.json
```

Evaluate:

```bash
python evaluate_submission.py \
  --model checkpoints/best_checkpoint.pth \
  --imgs-dir data/amtown_test_imgs \
  --masks-dir data/amtown_test_masks \
  --split-file data/amtown_test.txt \
  --scale 0.35 \
  --output output/training_report.json \
  --team YourTeam \
  --repo-url https://github.com/xxxxxx.git \
  --device auto \
  --history-file output/train_history.json \
  --train-split-file data/amtown_train.txt \
  --val-split-file data/amtown_val.txt
```

Generate plots:

```bash
python scripts/analyze_training.py \
  --report output/training_report.json \
  --history output/train_history.json \
  --figures-dir figures
```

## Training configuration (default pipeline)

The default settings in `scripts/run_pipeline.sh` (can be overridden via CLI args):

- **Epochs**: 30
- **Batch size**: 4
- **LR**: 1e-4
- **Scale**: 0.35
- **Optimizer**: AdamW
- **Scheduler**: Cosine annealing
- **Loss**: `ce_dice` (Cross-Entropy + Dice)
- **AMP**: enabled
- **Stable mode**: enabled

## Strategies used (policy)

This project includes several practical strategies to improve stability and performance:

- **Class imbalance mitigation**
  - Class-weighted Cross-Entropy (weights estimated from training subset)
  - WeightedRandomSampler (rebalance image sampling probability)
- **Online augmentation** (training only)
  - random flips, rotations, random resized crop, brightness/contrast/saturation jitter
  - a safety resize step to keep tensor shapes consistent after 90° rotations on non-square images
- **Robust execution**
  - Auto device selection: CUDA → MPS → CPU
  - OOM fallback logic (retry with safer batch/scale/AMP)
  - Conservative backend knobs in stable mode (to reduce crash risk)

## Evaluation and reporting

`evaluate_submission.py` evaluates a checkpoint on a split file, computes:

- Dice (mean over valid classes)
- mIoU (mean over valid classes)
- FWIoU (frequency weighted IoU)
- Pixel Accuracy and Mean Accuracy

It exports a standardized JSON report to `output/training_report.json`, including:

- `training_summary`
- `test_metrics`
- `per_class_results`
- `class_mapping`

## Generated figures and artifacts

After running the pipeline, you should see:

- **Figures** (from `scripts/analyze_training.py`)
  - `figures/training_loss_curve.png`
  - `figures/class_distribution.png`
  - `figures/per_class_iou.png`
  - `figures/summary_dashboard.png`
- **Outputs**
  - `output/train_history.json`
  - `output/training_report.json`
  - `output/AMtown02_submission_auto.json`
- **Checkpoints**
  - `checkpoints/checkpoint_epoch*.pth`
  - `checkpoints/best_checkpoint.pth`
- **Markdown report**
  - `docs/PROJECT_REPORT.md`

## Installation

```bash
python -m pip install -r requirements.txt
```



