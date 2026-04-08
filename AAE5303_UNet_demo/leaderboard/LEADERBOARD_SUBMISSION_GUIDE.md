# AAE5303 - Leaderboard Submission Guide

## 📁 Evaluation Dataset

**UAVScenes AMtown02 (interval=5)** - 2D Semantic Segmentation

| Resource | Link |
|----------|------|
| UAVScenes GitHub | https://github.com/sijieaaa/UAVScenes |
| MARS-LVIG Dataset | https://mars.hku.hk/dataset.html |

---

## 📊 Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| **Dice Score** | ↑ Higher is better | F1-Score for segmentation (0-100%) |
| **mIoU** | ↑ Higher is better | Mean Intersection over Union (0-100%) |
| **FWIoU** | ↑ Higher is better | Frequency Weighted IoU (0-100%) |

---

## 📄 JSON Submission Format

Submit your results using the following JSON format:

```json
{
    "group_name": "Team Alpha",
    "project_private_repo_url": "https://github.com/xxxxxx.git",
    "metrics": {
        "dice_score": 38.54,
        "miou": 32.93,
        "fwiou": 65.21
    }
}
```

### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `group_name` | string | Your group/team name | `"Team Alpha"` |
| `project_private_repo_url` | string | Your private GitHub repo URL | `"https://github.com/yourusername/project.git"` |
| `metrics.dice_score` | number | Dice Score (%) | `38.54` |
| `metrics.miou` | number | mIoU value (%) | `32.93` |
| `metrics.fwiou` | number | FWIoU value (%) | `65.21` |

### File Naming

`{YourGroupName}_leaderboard.json`

Example: `TeamAlpha_leaderboard.json`

---

## 🔧 How to Generate Submission

### Step 1: Train Your Model

```bash
python train.py \
    --epochs 20 \
    --batch-size 2 \
    --learning-rate 0.0001 \
    --scale 0.25
```

### Step 2: Run Evaluation Script

```bash
python evaluate_submission.py \
    --model checkpoints/checkpoint_epoch20.pth \
    --scale 0.25 \
    --output TeamAlpha_leaderboard.json \
    --team "Team Alpha" \
    --repo-url "https://github.com/yourusername/project.git"
```

### Step 3: Verify JSON Format

Ensure your JSON file matches the required format:

```python
import json

with open('TeamAlpha_leaderboard.json', 'r') as f:
    submission = json.load(f)

# Required fields
assert 'group_name' in submission, "Missing 'group_name'"
assert 'project_private_repo_url' in submission, "Missing 'project_private_repo_url'"
assert 'metrics' in submission, "Missing 'metrics'"
assert 'dice_score' in submission['metrics'], "Missing 'dice_score'"
assert 'miou' in submission['metrics'], "Missing 'miou'"
assert 'fwiou' in submission['metrics'], "Missing 'fwiou'"

# Validate URL format
assert submission['project_private_repo_url'].startswith('https://github.com/'), "Invalid GitHub URL"
assert submission['project_private_repo_url'].endswith('.git'), "URL should end with .git"

print("✓ Submission format is valid!")
print(f"Group: {submission['group_name']}")
print(f"mIoU: {submission['metrics']['miou']}%")
```

---

## 📊 Baseline Results

| Metric | Baseline Value |
|--------|----------------|
| **Dice Score** | 39.80% |
| **mIoU** | 72.73% |
| **FWIoU** | 88.85% |

**Training Configuration:**
- Epochs: 100 (best at epoch 65)
- Batch Size: 1
- Learning Rate: 5e-5
- Scale: 0.25
- Optimizer: AdamW
- Hardware: NVIDIA RTX 3060 6GB
- **Classes**: 14 valid classes (excluded unnamed/missing classes)

---

## 💡 Tips for Improvement

### 🚨 CRITICAL: Check Your Classes First!

**Before training**, verify:
1. ✅ Analyze your dataset to find actual class IDs
2. ✅ Exclude classes with empty names in `cmap.py`
3. ✅ Don't use classes that don't exist in your data
4. ✅ Use only valid, named classes present in dataset

**Wrong classes = Low mIoU (< 10%)!** See `TRAINING_TIPS.md` for details.

### Easy (Expected: +10-20% mIoU)
1. Train more epochs (15-20)
2. Adjust learning rate
3. Increase image scale (0.3-0.5)

### Medium (Expected: +20-30% mIoU)
4. Data augmentation (flip, rotate, color jitter)
5. Learning rate scheduler
6. Weighted loss for class imbalance

### Advanced (Expected: +30-40% mIoU)
7. Focal loss
8. Class-balanced sampling
9. Test-time augmentation
10. Ensemble methods

---

## 🌐 Leaderboard Website

**Live Rankings**: [https://qian9921.github.io/leaderboard_web/](https://qian9921.github.io/leaderboard_web/)

> View real-time leaderboard rankings and submit your results online!

### Current Baseline
- **mIoU**: 72.73% (Instructor Baseline)
- **Dice Score**: 39.80%
- **FWIoU**: 88.85%

Beat this baseline to appear on the leaderboard! 🏆
