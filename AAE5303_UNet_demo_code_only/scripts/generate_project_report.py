#!/usr/bin/env python3
"""Generate a Chinese project report markdown from report JSON files."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", default="output/training_report.json")
    parser.add_argument("--history-file", default="output/train_history.json")
    parser.add_argument("--output", default="docs/PROJECT_REPORT.md")
    parser.add_argument("--project-name", default="AAE5303 UNet Segmentation Pipeline")
    args = parser.parse_args()

    report = json.loads(Path(args.training_report).read_text())
    history = {}
    hist_path = Path(args.history_file)
    if hist_path.exists():
        history = json.loads(hist_path.read_text())

    ts = report["training_summary"]
    tm = report["test_metrics"]
    per_class = report["per_class_results"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# 项目说明报告

## 1. 项目概述
- 项目名称：{args.project_name}
- 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 任务类型：语义分割（UNet）
- 数据划分：Train/Val/Test = {ts["train_images"]}/{ts["val_images"]}/{ts["test_images"]}
- 类别数量：{ts["num_classes"]}

## 2. 训练启动方式
默认一键流程（训练 + 评估 + 可视化 + 报告）：

```bash
bash scripts/run_pipeline.sh
```

仅训练：

```bash
bash scripts/run_pipeline.sh train --epochs {ts["total_epochs"]} --batch-size {ts["batch_size"]} --scale {ts["image_scale"]}
```

仅评估：

```bash
bash scripts/run_pipeline.sh eval --model checkpoints/checkpoint_epoch{ts["total_epochs"]}.pth
```

## 3. 实现方式（详细）
- 模型：UNet 编码器-解码器结构，输出通道数为类别数。
- 数据：读取 `data/imgs` 和 `data/masks`，按同名文件对齐。
- 损失函数：CrossEntropy + Dice Loss。
- 优化器：{ts["optimizer"]}，学习率 {ts["learning_rate"]}。
- 设备策略：`--device auto`，优先 GPU，不可用时自动回退 CPU。
- 指标评估：Dice、mIoU、FWIoU、Pixel Accuracy、Mean Accuracy。

## 4. 输出文件说明
- `checkpoints/checkpoint_epoch*.pth`：每个 epoch 的模型权重。
- `output/train_history.json`：训练历史（epoch、loss、val_dice）。
- `output/training_report.json`：最终报告（与项目模板结构一致）。
- `figures/training_loss_curve.png`：训练曲线。
- `figures/class_distribution.png`：类别频率分布图。
- `figures/per_class_iou.png`：各类别 IoU / Dice 图。
- `figures/summary_dashboard.png`：综合看板。
- `docs/PROJECT_REPORT.md`：本项目说明报告。

## 5. 可视化信息
- 训练曲线来源：`train_history.json` 的 `epochs/train_loss/val_dice`。
- 指标图来源：`training_report.json` 的 `test_metrics/per_class_results`。
- 图表生成命令：

```bash
python scripts/analyze_training.py --report output/training_report.json --history output/train_history.json --figures-dir figures
```

## 6. 当前结果摘要
- Dice Score：{tm["dice_score"]}%
- mIoU：{tm["miou"]}%
- FWIoU：{tm["fwiou"]}%
- Pixel Accuracy：{tm["pixel_accuracy"]}%
- Mean Accuracy：{tm["mean_accuracy"]}%

## 7. 各类别表现
"""

    for cls, v in per_class.items():
        content += f"- {cls}: IoU={v['iou']}%, Dice={v['dice']}%, Frequency={v['frequency']}%\n"

    if history.get("epochs"):
        content += "\n## 8. 训练历史摘要\n"
        content += f"- 训练 epoch 数：{len(history.get('epochs', []))}\n"
        if history.get("train_loss"):
            content += f"- 最终 train loss：{history['train_loss'][-1]:.4f}\n"
        val_dice = [x for x in history.get("val_dice", []) if x is not None]
        if val_dice:
            content += f"- 最终 val dice：{val_dice[-1]:.4f}\n"

    content += """

## 9. 提升步骤实现方式（六项）
- 类别加权损失：统计训练集类别频率，构造 `class_weights`，用于 `CrossEntropy(weight=class_weights)`，并与 `Dice loss` 组合（`CE + Dice` 或 `Focal + Dice`）。
- 数据增强：仅训练阶段启用在线增强，包括随机翻转、随机旋转、随机裁剪缩放、亮度/对比度/饱和度扰动；验证与测试不做增强。
- 采样策略：训练集使用 `WeightedRandomSampler`，提高低频类别样本被采样概率，缓解类别不均衡。
- 优化器：默认 `AdamW`，可切换 `RMSprop`。
- 学习率策略：默认 `CosineAnnealingLR`，可切换 `ReduceLROnPlateau`、`OneCycleLR` 或 `none`。
- checkpoint 筛选：每个 epoch 保存 `checkpoint_epoch*.pth`，并基于验证集 Dice 更新 `best_checkpoint.pth`。

## 10. 对比实验运行方式
### 10.1 实验原则
- 固定数据划分、随机种子、训练轮数与评估脚本。
- 每次只变更 1-2 个因素，便于归因。
- 统一记录 Dice、mIoU、FWIoU、Pixel Accuracy、Mean Accuracy。

### 10.2 推荐实验矩阵
- 实验 A（Baseline）：关闭加权损失、增强、采样。
- 实验 B（+Weighted Loss）：仅开启类别加权损失。
- 实验 C（+Augmentation）：在 B 基础上开启增强。
- 实验 D（+Sampler）：在 C 基础上开启加权采样。
- 实验 E（+Optimizer/Scheduler）：在 D 基础上切换 AdamW + Cosine。
- 实验 F（Final）：启用全部策略并使用 best checkpoint。

### 10.3 运行命令（模板）
实验 A（Baseline）：
```bash
bash scripts/run_pipeline.sh train \
  --epochs 30 --batch-size 1 --scale 0.25 \
  --no-weighted-loss --no-augmentation --no-weighted-sampler \
  --optimizer rmsprop --scheduler none
```

实验 B（+Weighted Loss）：
```bash
bash scripts/run_pipeline.sh train \
  --epochs 30 --batch-size 1 --scale 0.25 \
  --weighted-loss --no-augmentation --no-weighted-sampler \
  --optimizer rmsprop --scheduler none
```

实验 C（+Augmentation）：
```bash
bash scripts/run_pipeline.sh train \
  --epochs 30 --batch-size 1 --scale 0.25 \
  --weighted-loss --augmentation --no-weighted-sampler \
  --optimizer rmsprop --scheduler none
```

实验 D（+Sampler）：
```bash
bash scripts/run_pipeline.sh train \
  --epochs 30 --batch-size 1 --scale 0.25 \
  --weighted-loss --augmentation --weighted-sampler \
  --optimizer rmsprop --scheduler none
```

实验 E（+Optimizer/Scheduler）：
```bash
bash scripts/run_pipeline.sh train \
  --epochs 30 --batch-size 1 --scale 0.25 \
  --weighted-loss --augmentation --weighted-sampler \
  --optimizer adamw --scheduler cosine
```

实验 F（Final）：
```bash
bash scripts/run_pipeline.sh
```

统一评估（示例）：
```bash
bash scripts/run_pipeline.sh eval --model checkpoints/best_checkpoint.pth
```

### 10.4 结果记录建议
- 建议整理对比表：实验编号、是否加权损失、是否增强、是否采样、优化器、调度器、best epoch、Dice、mIoU、FWIoU。
- 用同一份评估脚本导出 JSON，确保口径一致。
"""

    out_path.write_text(content)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
