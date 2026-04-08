#!/usr/bin/env python3
"""Generate visualization figures from output/training_report.json and output/train_history.json."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Configure matplotlib for better aesthetics
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass  # Use default style

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color scheme - professional academic palette
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#28A745',      # Green
    'danger': '#DC3545',       # Red
    'background': '#F5F5F5',
    'grid': '#E0E0E0'
}

def load_inputs(report_path: Path, history_path: Path):
    report = json.loads(report_path.read_text())
    history = {"epochs": [], "train_loss": [], "val_dice": []}
    if history_path.exists():
        history = json.loads(history_path.read_text())

    class_info = report["per_class_results"]
    training_data = {
        "epochs": history.get("epochs", []),
        "train_loss": history.get("train_loss", []),
        "val_dice": history.get("val_dice", []),
    }
    if not training_data["epochs"]:
        training_data = {"epochs": [1], "train_loss": [report["training_summary"]["final_train_loss"]], "val_dice": [report["training_summary"]["final_val_dice"]]}
    return report, class_info, training_data


def plot_training_loss_curve(training_data, output_path: str):
    """Generate training loss and validation Dice curve."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    epochs = training_data['epochs']
    train_loss = training_data['train_loss']
    val_dice = [0 if v is None else v for v in training_data['val_dice']]
    
    # Plot training loss
    color1 = COLORS['primary']
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color1, fontweight='bold')
    line1 = ax1.plot(epochs, train_loss, color=color1, linewidth=2.5, 
                     marker='o', markersize=8, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.0)
    
    # Plot validation Dice on secondary axis
    ax2 = ax1.twinx()
    color2 = COLORS['success']
    ax2.set_ylabel('Validation Dice Score', color=color2, fontweight='bold')
    line2 = ax2.plot(epochs, val_dice, color=color2, linewidth=2.5,
                     marker='s', markersize=8, label='Val Dice Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.5, 1.0)
    
    # Title and legend
    ax1.set_title('UNet Training Progress - UAVScenes Dataset', fontweight='bold', fontsize=14)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Add statistics text box
    stats_text = 'Training Statistics:\n'
    stats_text += f'• Initial Loss: {train_loss[0]:.4f}\n'
    stats_text += f'• Final Loss: {train_loss[-1]:.4f}\n'
    stats_text += f'• Loss Reduction: {((train_loss[0]-train_loss[-1])/train_loss[0]*100):.1f}%\n'
    stats_text += f'• Final Val Dice: {val_dice[-1]:.4f}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['primary'])
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_class_distribution(class_info, output_path: str):
    """Generate class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(class_info.keys())
    frequencies = [class_info[c]['frequency'] for c in classes]
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    bars = ax.bar(classes, frequencies, color=colors, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Semantic Class', fontweight='bold')
    ax.set_ylabel('Pixel Frequency (%)', fontweight='bold')
    ax.set_title('UAVScenes Class Distribution - Pixel Frequency', fontweight='bold', fontsize=14)
    
    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{freq:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add imbalance annotation
    ax.axhline(y=np.mean(frequencies), color=COLORS['secondary'], linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(frequencies):.2f}%')
    
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add statistics text
    stats_text = f'Class Imbalance Analysis:\n'
    stats_text += f'• Max: {max(frequencies):.2f}% (river)\n'
    stats_text += f'• Min: {min(frequencies):.2f}% (sedan)\n'
    stats_text += f'• Ratio: {max(frequencies)/min(frequencies):.0f}:1'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['secondary'])
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_class_iou(class_info, report, output_path: str):
    """Generate per-class IoU bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = list(class_info.keys())
    ious = [class_info[c]['iou'] for c in classes]
    dices = [class_info[c]['dice'] for c in classes]
    
    # Color bars based on performance
    iou_colors = [COLORS['success'] if iou > 50 else COLORS['accent'] if iou > 20 else COLORS['danger'] 
                  for iou in ious]
    
    # IoU chart
    ax1 = axes[0]
    bars1 = ax1.bar(classes, ious, color=iou_colors, edgecolor='white', linewidth=2)
    ax1.set_xlabel('Semantic Class', fontweight='bold')
    ax1.set_ylabel('IoU (%)', fontweight='bold')
    ax1.set_title('Per-Class IoU Performance', fontweight='bold')
    miou = report["test_metrics"]["miou"]
    mdice = report["test_metrics"]["dice_score"]
    ax1.axhline(y=miou, color=COLORS['secondary'], linestyle='--', 
                linewidth=2, label=f'mIoU: {miou:.2f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    plt.sca(ax1)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, iou in zip(bars1, ious):
        if iou > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{iou:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Dice chart
    ax2 = axes[1]
    dice_colors = [COLORS['success'] if dice > 60 else COLORS['accent'] if dice > 30 else COLORS['danger'] 
                   for dice in dices]
    bars2 = ax2.bar(classes, dices, color=dice_colors, edgecolor='white', linewidth=2)
    ax2.set_xlabel('Semantic Class', fontweight='bold')
    ax2.set_ylabel('Dice Score (%)', fontweight='bold')
    ax2.set_title('Per-Class Dice Score Performance', fontweight='bold')
    ax2.axhline(y=mdice, color=COLORS['secondary'], linestyle='--', 
                linewidth=2, label=f'mDice: {mdice:.2f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    plt.sca(ax2)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, dice in zip(bars2, dices):
        if dice > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{dice:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_dashboard(class_info, training_data, report, output_path: str):
    """Generate a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('AAE5303 Assignment 3: UNet Semantic Segmentation - UAVScenes Dataset\nTraining Summary Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Training curves (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = training_data['epochs']
    ax1.plot(epochs, training_data['train_loss'], color=COLORS['primary'], 
             linewidth=2.5, marker='o', label='Train Loss')
    val_dice = [0 if v is None else v for v in training_data['val_dice']]
    ax1.plot(epochs, val_dice, color=COLORS['success'], 
             linewidth=2.5, marker='s', label='Val Dice')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Value')
    ax1.set_title('Training Progress', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # 2. Class distribution (spans 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    classes = list(class_info.keys())
    frequencies = [class_info[c]['frequency'] for c in classes]
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    ax2.bar(classes, frequencies, color=colors, edgecolor='white')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Frequency (%)')
    ax2.set_title('Class Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.sca(ax2)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    # 3. Key Metrics Cards (middle row)
    tm = report["test_metrics"]
    metrics = [
        ('Dice Score', f'{tm["dice_score"]:.2f}%', COLORS['primary']),
        ('mIoU', f'{tm["miou"]:.2f}%', COLORS['secondary']),
        ('FWIoU', f'{tm["fwiou"]:.2f}%', COLORS['accent']),
        ('Pixel Acc', f'{tm["pixel_accuracy"]:.2f}%', COLORS['success']),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # Draw card background
        card = mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                        boxstyle="round,pad=0.02,rounding_size=0.05",
                                        facecolor=color, alpha=0.15,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(card)
        
        # Add text
        ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=20, fontweight='bold', color=color)
        ax.text(0.5, 0.3, label, ha='center', va='center', fontsize=11, color='gray')
        
        ax.axis('off')
    
    # 4. Per-class IoU table
    ax_table = fig.add_subplot(gs[2, :2])
    ax_table.axis('off')
    
    table_data = [['Class', 'IoU', 'Dice', 'Freq']]
    for cls, item in class_info.items():
        table_data.append([cls, f'{item["iou"]:.2f}%', f'{item["dice"]:.2f}%', f'{item["frequency"]:.2f}%'])
    table_data = table_data[:7] if len(table_data) > 7 else table_data
    
    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                           loc='center', cellLoc='center',
                           colColours=[COLORS['primary']]*4,
                           colWidths=[0.25, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax_table.set_title('Per-Class Results', fontweight='bold', pad=20)
    
    # 5. Configuration info
    ax_config = fig.add_subplot(gs[2, 2:])
    ax_config.axis('off')
    
    ts = report["training_summary"]
    config_text = f"""
    Training Configuration:
    ─────────────────────────────
    • Framework: PyTorch-UNet
    • Device: Auto (GPU/CPU)
    • Epochs: {ts["total_epochs"]}
    • Batch Size: {ts["batch_size"]}
    • Learning Rate: {ts["learning_rate"]}
    • Optimizer: {ts["optimizer"]}
    • Image Scale: {ts["image_scale"]}
    • Loss: CrossEntropy + Dice
    
    Dataset: HKisland_GNSS01
    ─────────────────────────────
    • Total Images: {ts["total_images"]}
    • Train/Val/Test: {ts["train_images"]}/{ts["val_images"]}/{ts["test_images"]}
    • Classes: {ts["num_classes"]}
    """
    
    ax_config.text(0.1, 0.95, config_text, transform=ax_config.transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['primary']))
    ax_config.set_title('Configuration', fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main function to run all analysis and generate visualizations."""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(Path(__file__).parent.parent / "output" / "training_report.json"))
    parser.add_argument("--history", default=str(Path(__file__).parent.parent / "output" / "train_history.json"))
    parser.add_argument("--figures-dir", default=str(Path(__file__).parent.parent / "figures"))
    args = parser.parse_args()

    report_path = Path(args.report)
    history_path = Path(args.history)
    report, class_info, training_data = load_inputs(report_path, history_path)

    # Paths
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AAE5303 Assignment 3 - Training Analysis")
    print("UNet Semantic Segmentation on UAVScenes Dataset")
    print("=" * 60)
    
    # Generate visualizations
    print("\n[1/4] Generating training loss curve...")
    plot_training_loss_curve(training_data, str(figures_dir / 'training_loss_curve.png'))
    
    print("\n[2/4] Generating class distribution...")
    plot_class_distribution(class_info, str(figures_dir / 'class_distribution.png'))
    
    print("\n[3/4] Generating per-class IoU analysis...")
    plot_per_class_iou(class_info, report, str(figures_dir / 'per_class_iou.png'))
    
    print("\n[4/4] Generating summary dashboard...")
    plot_summary_dashboard(class_info, training_data, report, str(figures_dir / 'summary_dashboard.png'))
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nFigures saved to: {figures_dir}")
    print("\nTest Set Metrics:")
    print(f"  • Dice Score: {report['test_metrics']['dice_score']:.2f}%")
    print(f"  • mIoU: {report['test_metrics']['miou']:.2f}%")
    print(f"  • FWIoU: {report['test_metrics']['fwiou']:.2f}%")


if __name__ == '__main__':
    main()
