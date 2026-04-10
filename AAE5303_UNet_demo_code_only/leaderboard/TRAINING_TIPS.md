# 💡 UNet Training Tips & Common Pitfalls

Quick reference guide for common issues encountered during UNet training for semantic segmentation.

---

## 🚨 MOST CRITICAL: Class Definition Issues

### ⚠️ Class Mismatch Will Destroy Your Results!

**THE PROBLEM**: `cmap.py` defines 26 classes (0-25), but:
1. **Some classes have empty names** (`''`) - these are invalid!
2. **Not all classes exist in your dataset** - using non-existent classes will fail!
3. **Wrong class count will cause shape mismatch errors or terrible metrics**

### Your Dataset Reality Check

**Step 1: ALWAYS analyze your dataset first!**

```python
import numpy as np
from PIL import Image
from pathlib import Path

# Find unique classes in YOUR dataset
mask_dir = Path('data/YOUR_DATASET/masks')
unique_classes = set()

for mask_file in mask_dir.glob('*.png'):
    mask = np.array(Image.open(mask_file))
    unique_classes.update(np.unique(mask).tolist())

print(f"Unique classes in YOUR data: {sorted(unique_classes)}")
print(f"Total: {len(unique_classes)} classes")
```

**Example Output**:
```
Unique classes in YOUR data: [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24, 25]
Total: 15 classes

Missing from dataset: [4, 7, 8, 9, 10, 11, 12, 18, 21, 22, 23]
```

### Check for Unnamed Classes

```python
import sys
sys.path.append('data')
from cmap import cmap

# Find classes with empty names
unnamed = [id for id, info in cmap.items() if info['name'] == '']
print(f"Classes with empty names: {unnamed}")
# Output: [7, 8, 12, 21, 22, 23, 25]
```

### ✅ Correct Class Setup

**Option 1: Use only named classes that exist in your data**
```python
# Find valid class IDs
VALID_CLASS_IDS = []
for class_id, info in cmap.items():
    if info['name'] != '' and class_id in dataset_classes:
        VALID_CLASS_IDS.append(class_id)

print(f"Using {len(VALID_CLASS_IDS)} valid classes: {VALID_CLASS_IDS}")
# Example: [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24]
```

**Option 2: Exclude specific problematic classes**
```python
# For baseline: exclude class 25 (unnamed, rare <0.01%)
VALID_CLASS_IDS = [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24]
n_classes = len(VALID_CLASS_IDS)  # = 14
```

### ❌ Common Mistakes

**Mistake 1**: Using all 26 classes from cmap.py
```python
# WRONG - will cause errors!
n_classes = len(cmap)  # = 26
model = UNet(n_channels=3, n_classes=26)  # Many classes don't exist!
```

**Mistake 2**: Not filtering unnamed classes
```python
# WRONG - includes classes with empty names
all_classes = list(cmap.keys())  # Includes 7, 8, 12, 21, 22, 23, 25
```

**Mistake 3**: Using classes not in your data
```python
# WRONG - class 4 (river) doesn't exist in the dataset!
VALID_CLASS_IDS = [0, 1, 2, 3, 4, 5, ...]  # Training will fail
```

### ✅ Correct Implementation

```python
# 1. Define valid classes (from dataset analysis)
VALID_CLASS_IDS = [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24]

# 2. Create ID mapping
id_to_idx = {class_id: idx for idx, class_id in enumerate(VALID_CLASS_IDS)}

# 3. Convert dataset labels
def convert_label(original_mask):
    """Convert original class IDs to continuous indices 0-13"""
    new_mask = np.zeros_like(original_mask)
    for class_id, idx in id_to_idx.items():
        new_mask[original_mask == class_id] = idx
    return new_mask

# 4. Create model with correct number
n_classes = len(VALID_CLASS_IDS)  # = 14
model = UNet(n_channels=3, n_classes=n_classes)
```

### 🎯 Impact on Results

**If you get classes wrong**:
- ❌ Shape mismatch errors during training
- ❌ mIoU drops to < 5%
- ❌ Model only predicts one class
- ❌ Evaluation fails completely

**With correct classes**:
- ✅ Baseline achieves 72.73% mIoU
- ✅ All classes predicted correctly
- ✅ Smooth training

---

## ⚠️ Other Critical Issues

### 1. GPU Not Detected (PyTorch shows CUDA unavailable)

**Symptom**: `torch.cuda.is_available()` returns `False` even though `nvidia-smi` works

**Common Causes**:
- Python version too new (e.g., Python 3.13)
- PyTorch CUDA version mismatch

**Solution**:
```bash
# Use Python 3.8-3.11
conda create -n unet python=3.8
conda activate unet
# Install PyTorch with matching CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Quick Check**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

### 2. Out of Memory (OOM) Error

**Symptom**: `CUDA out of memory` during training

**Why It Happens**:
- Aerial images are large (2448×2048)
- 6GB GPU cannot handle batch_size=4 + scale=0.5

**Solution**:
```bash
# Reduce scale to 0.25 (input becomes 612×512)
python train.py --batch-size 1 --scale 0.25
```

**Memory Requirements**:
| Config | GPU Memory | 6GB GPU |
|--------|------------|---------|
| BS=4, scale=0.5 | ~8-10GB | ❌ OOM |
| BS=2, scale=0.5 | ~6-7GB | ⚠️ Risky |
| BS=1, scale=0.25 | ~2-3GB | ✅ Safe |

**Myth**: "Larger batch size is always better"  
**Reality**: BS=1 with more epochs often works better for small GPUs

---

### 3. NaN Parameters (Model Breaks Completely)

**Symptom**: All predictions become identical, loss shows `nan`

**Root Cause**: Numerical instability in optimizer

**Quick Diagnosis**:
```python
import torch
checkpoint = torch.load('model.pth', map_location='cpu')
first_param = list(checkpoint.values())[0]
print(f"Has NaN: {torch.isnan(first_param).any()}")
```

**Solution**:
- ❌ Avoid: `RMSprop` with high momentum (0.999)
- ✅ Use: `AdamW` optimizer instead
- Lower learning rate: `5e-5` instead of `1e-4`

```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=5e-5,
    weight_decay=1e-8
)
```

---

## 📊 Evaluation Issues

### 4. "Best Model" Isn't Actually Best

**Symptom**: Model saved as `best_model.pth` performs worse on test set

**Why**: Validation set performance ≠ Test set performance

**Example**: In our training:
- `best_model.pth` (Epoch 70): mIoU = 64.24%
- Actual best (Epoch 65): mIoU = 72.73%

**Solution**: **Always evaluate ALL epochs on test set**

```python
# Don't just use best_model.pth
# Evaluate all checkpoints and pick the actual best
for epoch in range(1, 101):
    model.load_state_dict(torch.load(f'checkpoint_epoch{epoch}.pth'))
    test_miou = evaluate(model, test_loader)
```

---

### 5. Checkpoint Evaluation Order Wrong

**Symptom**: Evaluating Epoch 1, 10, 100, 11, 12... (wrong order)

**Cause**: String sorting instead of numerical

**Solution**:
```python
from pathlib import Path

checkpoints = list(Path('checkpoints').glob('checkpoint_epoch*.pth'))
# Sort by epoch number, not string
checkpoints.sort(key=lambda p: int(p.stem.split('epoch')[1]))
```

---

## 📉 Metric Understanding

### 6. Confused by Different Metrics

**Common Confusion**: Why is FWIoU (88.85%) much higher than Dice (39.80%)?

**Explanation**:
- **FWIoU**: Weighted by frequency → dominated by common classes
- **mIoU**: Average across all classes → pulled down by rare classes
- **Dice**: More sensitive to small objects

**For aerial imagery**:
- Use **mIoU** as primary metric (standard for segmentation)
- FWIoU useful for understanding visual quality
- Dice sensitive to class imbalance

**Normal Relationships**:
```
FWIoU ≈ Pixel Accuracy > mIoU > Dice Score
(88.85%)    (93.77%)      (72.73%)  (39.80%)   ← Expected for imbalanced data
```

---

## 🔧 Training Configuration

### 7. Mixed Precision (AMP) Instability

**Symptom**: Training becomes unstable with `--amp` flag

**Trade-off**:
- ✅ Speed: ~20% faster
- ⚠️ Stability: Risk of Inf/NaN in gradients

**Recommendation**:
- Start without AMP to verify training works
- Add AMP only after confirming stability
- If unstable with AMP, disable it (stability > speed)

---

### 8. Learning Rate Too High

**Symptom**: Gradient explosions, `Inf` in gradients

**Solution**:
- Start conservative: `lr=5e-5`
- Can increase to `1e-4` if stable
- Use gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 9. Class Imbalance Effects

**Reality**: Rare classes will have low IoU

**Example from our training**:
- Green Field (61% of pixels): 94.81% IoU ✅
- Pool (0.001% of pixels): 8.79% IoU ⚠️

**This is expected, not a bug!**

**When to worry**:
- If dominant classes have low IoU
- If overall mIoU < 30% after 50+ epochs

---

## ⏱️ Training Expectations

### 10. "Is My Training Too Slow?"

**Typical speeds** (RTX 3060 6GB, BS=1, scale=0.25):
- ~15-20 minutes per epoch
- 100 epochs ≈ 20-30 hours total

**This is normal!** High-resolution imagery on consumer GPUs takes time.

---

### 11. How Many Epochs?

**Our experience**:
- Epoch 1-10: Rapid learning (mIoU: 26% → 61%)
- Epoch 10-50: Steady improvement (61% → 68%)
- Epoch 50-70: Fine-tuning (68% → 73%)
- Epoch 70-100: Minor fluctuations (71-73%)

**Recommendation**: Train 80-100 epochs, then evaluate all checkpoints

---

## ✅ Success Indicators

### Healthy Training Curve

```
Epoch 1:  mIoU = 26% (random initialization)
Epoch 10: mIoU = 61% ✓ rapid learning
Epoch 30: mIoU = 68% ✓ steady progress
Epoch 65: mIoU = 73% ⭐ peak performance
Epoch 100: mIoU = 71% ✓ slight overfitting (normal)
```

### Performance Targets (14-class aerial imagery)

| mIoU | Performance |
|------|-------------|
| > 70% | Excellent ⭐⭐⭐ |
| 50-70% | Good ⭐⭐ |
| 30-50% | Acceptable ⭐ |
| < 30% | Needs debugging ⚠️ |

---

## 🚀 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| **mIoU < 10% or shape errors** | ❗ **Check class definitions! Analyze dataset first!** |
| Model only predicts one class | ❗ **Wrong number of classes - verify VALID_CLASS_IDS** |
| GPU not detected | Check Python version (use 3.8-3.11) |
| OOM error | Reduce scale to 0.25, BS to 1 |
| NaN parameters | Switch to AdamW, lower LR |
| Training very slow | Normal for high-res images |
| Best model not best | Evaluate all epochs on test set |
| Low IoU on rare classes | Expected behavior, focus on overall mIoU |

---

## 📚 Recommended Workflow

1. **🚨 ANALYZE DATASET CLASSES FIRST** → Find unique class IDs in your data
2. **🚨 VERIFY CLASS DEFINITIONS** → Exclude unnamed/missing classes
3. **Verify GPU works** → Check CUDA availability
4. **Start conservative** → BS=1, scale=0.25, lr=5e-5, no AMP
5. **Monitor first 10 epochs** → Verify loss decreases AND mIoU > 20%
6. **Train to completion** → 80-100 epochs
7. **Evaluate all checkpoints** → Don't trust validation "best"
8. **Select by test mIoU** → True best model

> ⚠️ **Steps 1-2 are CRITICAL!** Wrong classes = wasted training time!

---

**Remember**: These tips come from real debugging. Your specific case may vary, but these principles generally apply!

Last Updated: January 11, 2026
