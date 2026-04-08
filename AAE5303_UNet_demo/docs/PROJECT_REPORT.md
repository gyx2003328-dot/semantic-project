# 项目说明报告

## 1. 项目概述
- 项目名称：AAE5303 UNet Segmentation Pipeline
- 生成时间：2026-04-08 13:22:27
- 任务类型：语义分割（UNet）
- 数据划分：Train/Val/Test = 949/203/204
- 类别数量：8

## 2. 训练启动方式
默认一键流程（训练 + 评估 + 可视化 + 报告）：

```bash
bash scripts/run_pipeline.sh
```

仅训练：

```bash
bash scripts/run_pipeline.sh train --epochs 10 --batch-size 2 --scale 0.35
```

仅评估：

```bash
bash scripts/run_pipeline.sh eval --model checkpoints/checkpoint_epoch10.pth
```

## 3. 实现方式（详细）
- 模型：UNet 编码器-解码器结构，输出通道数为类别数。
- 数据：读取 `data/imgs` 和 `data/masks`，按同名文件对齐。
- 损失函数：CrossEntropy + Dice Loss。
- 优化器：adamw，学习率 0.0001。
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
- Dice Score：88.57%
- mIoU：80.24%
- FWIoU：77.44%
- Pixel Accuracy：86.46%
- Mean Accuracy：92.25%

## 7. 各类别表现
- ground: IoU=55.23%, Dice=71.16%, Frequency=20.22%
- roof: IoU=87.9%, Dice=93.56%, Frequency=1.83%
- building: IoU=87.85%, Dice=93.53%, Frequency=0.37%
- river: IoU=78.39%, Dice=87.89%, Frequency=31.07%
- road: IoU=84.83%, Dice=91.79%, Frequency=0.41%
- green_field: IoU=93.79%, Dice=96.79%, Frequency=18.1%
- wild_field: IoU=80.95%, Dice=89.47%, Frequency=27.94%
- sedan: IoU=72.99%, Dice=84.38%, Frequency=0.05%

## 8. 训练历史摘要
- 训练 epoch 数：10
- 最终 train loss：0.0948
- 最终 val dice：0.7460


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
bash scripts/run_pipeline.sh train   --epochs 30 --batch-size 1 --scale 0.25   --no-weighted-loss --no-augmentation --no-weighted-sampler   --optimizer rmsprop --scheduler none
```

实验 B（+Weighted Loss）：
```bash
bash scripts/run_pipeline.sh train   --epochs 30 --batch-size 1 --scale 0.25   --weighted-loss --no-augmentation --no-weighted-sampler   --optimizer rmsprop --scheduler none
```

实验 C（+Augmentation）：
```bash
bash scripts/run_pipeline.sh train   --epochs 30 --batch-size 1 --scale 0.25   --weighted-loss --augmentation --no-weighted-sampler   --optimizer rmsprop --scheduler none
```

实验 D（+Sampler）：
```bash
bash scripts/run_pipeline.sh train   --epochs 30 --batch-size 1 --scale 0.25   --weighted-loss --augmentation --weighted-sampler   --optimizer rmsprop --scheduler none
```

实验 E（+Optimizer/Scheduler）：
```bash
bash scripts/run_pipeline.sh train   --epochs 30 --batch-size 1 --scale 0.25   --weighted-loss --augmentation --weighted-sampler   --optimizer adamw --scheduler cosine
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
