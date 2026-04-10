# 🏆 AAE5303 Semantic Segmentation - Leaderboard

## 📁 Evaluation Dataset

**UAVScenes AMtown02 (interval=5)** - 2D Semantic Segmentation Dataset

| Resource | Link |
|----------|------|
| UAVScenes GitHub | https://github.com/sijieaaa/UAVScenes |
| MARS-LVIG Dataset | https://mars.hku.hk/dataset.html |

---

## 📊 Evaluation Metrics

The leaderboard evaluates submissions using three standard semantic segmentation metrics. All metrics are computed on the **AMtown02 interval=5** split.

---

### 1. Dice Score (F1-Score) ↑

**Higher is better** | Range: 0% to 100%

#### Definition

Dice Score (also known as F1-Score) measures the overlap between predicted segmentation and ground truth. It is the harmonic mean of precision and recall.

#### Mathematical Formula

$$Dice = \frac{2 \times TP}{2 \times TP + FP + FN} = \frac{2 \times |P \cap G|}{|P| + |G|}$$

where:
- $TP$ = True Positives (correctly predicted pixels)
- $FP$ = False Positives (incorrectly predicted as positive)
- $FN$ = False Negatives (missed positive pixels)
- $P$ = Predicted segmentation mask
- $G$ = Ground truth mask

For multi-class segmentation, mean Dice is computed:

$$mDice = \frac{1}{C}\sum_{c=1}^{C} Dice_c$$

#### Reference Code

```python
import numpy as np

def calculate_dice(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """
    Calculate mean Dice Score for semantic segmentation.
    
    Args:
        pred: Predicted mask, shape (H, W), dtype int, values in [0, num_classes-1]
        target: Ground truth mask, shape (H, W), dtype int, values in [0, num_classes-1]
        num_classes: Number of semantic classes
    
    Returns:
        Mean Dice Score (0 to 1)
    """
    dice_per_class = []
    
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        intersection = np.sum(pred_c * target_c)
        union = np.sum(pred_c) + np.sum(target_c)
        
        if union > 0:
            dice = 2 * intersection / union
            dice_per_class.append(dice)
    
    return np.mean(dice_per_class) if dice_per_class else 0.0
```

---

### 2. mIoU (Mean Intersection over Union) ↑

**Higher is better** | Range: 0% to 100%

#### Definition

IoU (Jaccard Index) measures the overlap ratio between prediction and ground truth. mIoU is the standard metric for semantic segmentation benchmarks (PASCAL VOC, Cityscapes, ADE20K).

#### Mathematical Formula

$$IoU = \frac{TP}{TP + FP + FN} = \frac{|P \cap G|}{|P \cup G|}$$

$$mIoU = \frac{1}{C}\sum_{c=1}^{C} IoU_c$$

where:
- $C$ = Number of classes
- $IoU_c$ = IoU for class $c$

#### Relationship with Dice

$$Dice = \frac{2 \times IoU}{1 + IoU}$$

$$IoU = \frac{Dice}{2 - Dice}$$

#### Reference Code

```python
import numpy as np

def calculate_miou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """
    Calculate mean IoU for semantic segmentation.
    
    Args:
        pred: Predicted mask, shape (H, W), dtype int
        target: Ground truth mask, shape (H, W), dtype int
        num_classes: Number of semantic classes
    
    Returns:
        mIoU value (0 to 1)
    """
    # Build confusion matrix
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    # Calculate IoU per class
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iou = intersection / (union + 1e-10)
    
    # Mean over valid classes
    valid = hist.sum(axis=1) > 0
    miou = np.mean(iou[valid])
    
    return miou
```

---

### 3. FWIoU (Frequency Weighted IoU) ↑

**Higher is better** | Range: 0% to 100%

#### Definition

FWIoU weights each class's IoU by its pixel frequency, giving more importance to common classes. It better reflects overall visual quality when class distribution is imbalanced.

#### Mathematical Formula

$$FWIoU = \frac{\sum_{c=1}^{C} freq_c \times IoU_c}{\sum_{c=1}^{C} freq_c}$$

where:
- $freq_c$ = Frequency of class $c$ (number of pixels / total pixels)
- $IoU_c$ = IoU for class $c$

#### Reference Code

```python
import numpy as np

def calculate_fwiou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """
    Calculate Frequency Weighted IoU.
    
    Args:
        pred: Predicted mask, shape (H, W), dtype int
        target: Ground truth mask, shape (H, W), dtype int
        num_classes: Number of semantic classes
    
    Returns:
        FWIoU value (0 to 1)
    """
    # Build confusion matrix
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    # Calculate frequency and IoU per class
    freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iou = intersection / (union + 1e-10)
    
    # Frequency weighted IoU
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
    
    return fwiou
```

---

## 📦 Complete Evaluation Script

Use this script to compute all three metrics for your submission:

```python
#!/usr/bin/env python3
"""
AAE5303 Leaderboard - Metrics Calculation Script
Semantic Segmentation on UAVScenes Dataset
"""

import numpy as np
import json
from pathlib import Path
from datetime import date

# Install required packages:
# pip install numpy torch pillow tqdm

def build_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Build confusion matrix from predictions and targets."""
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def calculate_all_metrics(confusion_matrix: np.ndarray) -> dict:
    """
    Calculate Dice, mIoU, and FWIoU from confusion matrix.
    
    Args:
        confusion_matrix: Shape (num_classes, num_classes)
    
    Returns:
        Dictionary with all metrics
    """
    hist = confusion_matrix
    
    # Per-class IoU
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iou = intersection / (union + 1e-10)
    
    # Per-class Dice
    dice = 2 * intersection / (hist.sum(axis=1) + hist.sum(axis=0) + 1e-10)
    
    # Frequency
    freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
    
    # Valid classes
    valid = hist.sum(axis=1) > 0
    
    # Compute mean metrics
    miou = np.mean(iou[valid])
    mdice = np.mean(dice[valid])
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
    
    return {
        'dice_score': round(float(mdice) * 100, 2),
        'miou': round(float(miou) * 100, 2),
        'fwiou': round(float(fwiou) * 100, 2)
    }

def generate_submission_json(group_name: str, repo_url: str, metrics: dict, output_path: str):
    """Generate submission JSON file."""
    submission = {
        "group_name": group_name,
        "project_private_repo_url": repo_url,
        "metrics": metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=4)
    
    print(f"Submission saved to: {output_path}")
    print(json.dumps(submission, indent=4))

# Example usage:
if __name__ == "__main__":
    # After running your model on test set and collecting predictions:
    # metrics = calculate_all_metrics(confusion_matrix)
    
    # Generate submission
    metrics = {
        "dice_score": 38.54,
        "miou": 32.93,
        "fwiou": 65.21
    }
    
    generate_submission_json(
        group_name="Team Alpha",
        repo_url="https://github.com/yourusername/project.git",
        metrics=metrics,
        output_path="TeamAlpha_leaderboard.json"
    )
```

---

## 📄 Submission Format

Submit a JSON file with the following format:

```json
{
    "group_name": "Team Alpha",
    "project_private_repo_url": "https://github.com/yourusername/project.git",
    "metrics": {
        "dice_score": 38.54,
        "miou": 32.93,
        "fwiou": 65.21
    }
}
```

**Required Fields:**
- `group_name`: Your team name
- `project_private_repo_url`: Your private GitHub repository URL (must end with `.git`)
- `metrics`: Performance metrics (Dice Score, mIoU, FWIoU in percentage)

**Template file**: [submission_template.json](./submission_template.json)

---

## 🏆 Current Leaderboard

| Rank | Group | Dice Score ↑ | mIoU ↑ | FWIoU ↑ | Date |
|------|-------|--------------|--------|---------|------|
| - | **Baseline (Instructor)** | 39.80% | **72.73%** | **88.85%** | 2026-01-09 |

### 🎯 Baseline Details

**Model**: UNet with AdamW optimizer (Epoch 65)
- **Configuration**: Batch Size=1, Scale=0.25, AdamW optimizer, lr=5e-5
- **Hardware**: NVIDIA RTX 3060 6GB (consumer GPU)
- **Training**: 100 epochs, ~26 hours total
- **Classes Used**: 14 valid classes (excluding unnamed classes)
- **Key Achievement**: 72.73% mIoU on aerial imagery segmentation

**Per-Class Performance** (Top 5):
- Green Field: 94.81% IoU
- Roof: 91.32% IoU  
- Solar Board: 83.52% IoU
- Background: 81.26% IoU
- Dirt Motor Road: 78.81% IoU

> **🎓 Your Goal**: Beat this baseline to appear on the leaderboard!

---

## 🌐 Leaderboard Website

**Live Rankings**: [https://qian9921.github.io/leaderboard_web/](https://qian9921.github.io/leaderboard_web/)

> View real-time leaderboard rankings and submit your results online!
