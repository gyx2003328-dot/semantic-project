import argparse
import json
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def configure_stable_runtime(stable_mode: bool):
    """Use conservative backend settings to reduce crash risk."""
    if not stable_mode:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Keep matmul precision conservative for stability
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class AugmentDataset(Dataset):
    """Apply online data augmentation on top of a base dataset/subset."""

    def __init__(self, dataset, enabled: bool = False):
        self.dataset = dataset
        self.enabled = enabled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        mask = sample['mask']
        target_h, target_w = mask.shape

        if not self.enabled:
            return sample

        # Geometric transforms (must be synchronized for image/mask)
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if random.random() < 0.2:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])
        if random.random() < 0.3:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])

        # Random resized crop
        h, w = mask.shape
        if min(h, w) > 64 and random.random() < 0.35:
            scale = random.uniform(0.8, 1.0)
            crop_h, crop_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            image = TF.resized_crop(
                image, top, left, crop_h, crop_w, [h, w],
                interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            mask = TF.resized_crop(
                mask.unsqueeze(0).float(), top, left, crop_h, crop_w, [h, w],
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0).long()

        # Photometric transforms (image only)
        if random.random() < 0.4:
            image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
            image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
            image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))
            image = torch.clamp(image, 0.0, 1.0)

        # Safety patch: always force a fixed output size for DataLoader stack.
        # This avoids HxW / WxH mismatch caused by 90/270-degree rotations on non-square images.
        if image.shape[1] != target_h or image.shape[2] != target_w:
            image = TF.resize(
                image, [target_h, target_w],
                interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            mask = TF.resize(
                mask.unsqueeze(0).float(), [target_h, target_w],
                interpolation=InterpolationMode.NEAREST
            ).squeeze(0).long()

        return {
            'image': image.contiguous(),
            'mask': mask.contiguous(),
        }


def estimate_class_stats(dataset, indices, n_classes):
    """Estimate class frequencies and sample weights from train subset."""
    pixel_counts = np.zeros(n_classes, dtype=np.float64)
    per_sample_hist = []

    for i in indices:
        sample = dataset[i]
        mask = sample['mask'].numpy()
        # Defensive slice: keep exactly n_classes bins to avoid shape mismatch.
        hist = np.bincount(mask.reshape(-1), minlength=n_classes).astype(np.float64)[:n_classes]
        pixel_counts += hist
        per_sample_hist.append(hist)

    class_freq = pixel_counts / max(pixel_counts.sum(), 1.0)
    inv = 1.0 / (class_freq + 1e-8)
    inv = inv / inv.mean()

    sample_weights = []
    for hist in per_sample_hist:
        present = np.where(hist > 0)[0]
        if len(present) == 0:
            sample_weights.append(1.0)
            continue
        sample_weights.append(float(inv[present].mean()))

    # Weighted CE class weights (sqrt dampens very extreme imbalance)
    class_weights = np.sqrt(inv)
    class_weights = class_weights / class_weights.mean()
    return class_freq, class_weights.astype(np.float32), sample_weights


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def resolve_device(device_arg: str) -> torch.device:
    """Auto-pick a usable device: CUDA -> MPS -> CPU."""
    if device_arg == 'cpu':
        return torch.device('cpu')

    if device_arg.startswith('cuda'):
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device='cuda')
                return torch.device(device_arg)
            except Exception as e:
                logging.warning(f'CUDA requested but not usable: {e}. Falling back to CPU.')
        else:
            logging.warning('CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.')
        return torch.device('cpu')

    # auto mode
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device='cuda')
            return torch.device('cuda')
        except Exception as e:
            logging.warning(f'CUDA detected but unusable: {e}.')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_wandb: bool = False,
        history_out: str = "output/train_history.json",
        use_weighted_loss: bool = True,
        use_augmentation: bool = True,
        use_weighted_sampler: bool = True,
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        min_lr: float = 1e-6,
        focal_gamma: float = 2.0,
        loss_name: str = "ce_dice",
        stable_mode: bool = True,
        num_workers: int = -1,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    dataset_n_classes = len(dataset.mask_values)
    if model.n_classes != dataset_n_classes:
        raise ValueError(
            f"类别数不一致：model.n_classes={model.n_classes}, "
            f"dataset classes={dataset_n_classes}, mask_values={dataset.mask_values}. "
            f"请将 --classes 设置为 {dataset_n_classes}。"
        )

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_indices = train_set.indices if hasattr(train_set, 'indices') else list(range(len(train_set)))
    class_freq, class_weights_np, sample_weights = estimate_class_stats(dataset, train_indices, model.n_classes)
    class_weights_t = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    if use_augmentation:
        train_set = AugmentDataset(train_set, enabled=True)

    # 3. Create data loaders
    cpu_count = os.cpu_count() or 1
    if num_workers < 0:
        resolved_workers = min(4, cpu_count) if stable_mode else min(16, cpu_count)
    else:
        resolved_workers = max(0, int(num_workers))
    resolved_pin_memory = bool(pin_memory and device.type == 'cuda')

    loader_args = dict(
        batch_size=batch_size,
        num_workers=resolved_workers,
        pin_memory=resolved_pin_memory,
    )
    if resolved_workers > 0:
        loader_args["persistent_workers"] = stable_mode
        loader_args["prefetch_factor"] = max(1, int(prefetch_factor))

    sampler = None
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
    train_loader = DataLoader(train_set, shuffle=(sampler is None), sampler=sampler, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Optional W&B logging (disabled by default for offline/air-gapped runs)
    experiment = None
    if use_wandb and wandb is not None:
        experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Stable mode:     {stable_mode}
        Num workers:     {resolved_workers}
        Pin memory:      {resolved_pin_memory}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif scheduler_name == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=max(len(train_loader), 1)
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if model.n_classes > 1:
        ce_weight = class_weights_t if use_weighted_loss else None
        if loss_name == "ce_dice":
            criterion = nn.CrossEntropyLoss(weight=ce_weight)
        elif loss_name == "focal_dice":
            criterion = FocalCrossEntropy(gamma=focal_gamma, weight=ce_weight)
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    best_val_score = -1.0
    best_epoch = 0
    history = {
        "epochs": [],
        "train_loss": [],
        "val_dice": [],
        "meta": {
            "train_images": n_train,
            "val_images": n_val,
            "total_images": len(dataset),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_scale": img_scale,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "weighted_loss": use_weighted_loss,
            "weighted_sampler": use_weighted_sampler,
            "augmentation": use_augmentation,
            "loss_name": loss_name,
            "class_freq": class_freq.tolist(),
            "class_weights": class_weights_np.tolist(),
            "stable_mode": stable_mode,
            "num_workers": resolved_workers,
            "pin_memory": resolved_pin_memory,
        },
    }

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        last_val_score = None
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                if scheduler is not None and scheduler_name == "onecycle":
                    scheduler.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if experiment is not None:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if experiment is not None and not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if experiment is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        last_val_score = float(val_score)
                        if scheduler is not None and scheduler_name == "plateau":
                            scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            if experiment is None:
                                continue
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except Exception:
                            pass

        avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
        if last_val_score is None:
            last_val_score = float(evaluate(model, val_loader, device, amp))
            if scheduler is not None and scheduler_name == "plateau":
                scheduler.step(last_val_score)
        if scheduler is not None and scheduler_name == "cosine":
            scheduler.step()

        history["epochs"].append(epoch)
        history["train_loss"].append(float(avg_epoch_loss))
        history["val_dice"].append(float(last_val_score) if last_val_score is not None else None)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        # Save best checkpoint by validation Dice
        if last_val_score is not None and float(last_val_score) > best_val_score:
            best_val_score = float(last_val_score)
            best_epoch = epoch
            best_state = model.state_dict()
            best_state['mask_values'] = dataset.mask_values
            torch.save(best_state, str(dir_checkpoint / 'best_checkpoint.pth'))
            logging.info(f'Best checkpoint updated at epoch {epoch} (val_dice={best_val_score:.4f})')

    history_path = Path(history_out)
    history["best"] = {"epoch": best_epoch, "val_dice": best_val_score}
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    logging.info(f"Training history saved to {history_path}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto|cuda|cpu')
    parser.add_argument('--wandb', action='store_true', default=False, help='Enable Weights & Biases logging')
    parser.add_argument('--history-out', type=str, default='output/train_history.json', help='Path to save training history JSON')
    parser.add_argument('--weighted-loss', action='store_true', default=True, help='Use class-weighted loss')
    parser.add_argument('--no-weighted-loss', action='store_false', dest='weighted_loss', help='Disable class-weighted loss')
    parser.add_argument('--augmentation', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation', help='Disable data augmentation')
    parser.add_argument('--weighted-sampler', action='store_true', default=True, help='Enable weighted random sampler')
    parser.add_argument('--no-weighted-sampler', action='store_false', dest='weighted_sampler', help='Disable weighted sampler')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'rmsprop'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'onecycle', 'none'], help='LR scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum LR for cosine scheduler')
    parser.add_argument('--loss-name', type=str, default='ce_dice', choices=['ce_dice', 'focal_dice'], help='Segmentation loss type')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma for focal loss')
    parser.add_argument('--stable-mode', action='store_true', default=True, help='Enable conservative stable settings')
    parser.add_argument('--no-stable-mode', action='store_false', dest='stable_mode', help='Disable conservative stable settings')
    parser.add_argument('--num-workers', type=int, default=-1, help='DataLoader workers (-1 means auto)')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch factor when workers>0')
    parser.add_argument('--pin-memory', action='store_true', default=True, help='Enable pin_memory when CUDA')
    parser.add_argument('--no-pin-memory', action='store_false', dest='pin_memory', help='Disable pin_memory')
    parser.add_argument('--img-dir', type=str, default='data/imgs', help='Training images directory')
    parser.add_argument('--mask-dir', type=str, default='data/masks', help='Training masks directory')

    return parser.parse_args()


def run_with_oom_fallback(model, device, args):
    """Retry with safer batch/scale when OOM happens."""
    attempts = [
        (args.batch_size, args.scale, args.amp),
        (max(1, args.batch_size // 2), args.scale, True),
        (1, min(args.scale, 0.35), True),
        (1, 0.25, False),
    ]
    seen = set()
    for idx, (bs, sc, amp_flag) in enumerate(attempts, start=1):
        key = (bs, sc, amp_flag)
        if key in seen:
            continue
        seen.add(key)
        logging.info(f"[Attempt {idx}] batch_size={bs}, scale={sc}, amp={amp_flag}")
        try:
            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=bs,
                learning_rate=args.lr,
                device=device,
                img_scale=sc,
                val_percent=args.val / 100,
                amp=amp_flag,
                use_wandb=args.wandb,
                history_out=args.history_out,
                use_weighted_loss=args.weighted_loss,
                use_augmentation=args.augmentation,
                use_weighted_sampler=args.weighted_sampler,
                optimizer_name=args.optimizer,
                scheduler_name=args.scheduler,
                min_lr=args.min_lr,
                focal_gamma=args.focal_gamma,
                loss_name=args.loss_name,
                stable_mode=args.stable_mode,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
            )
            return
        except torch.cuda.OutOfMemoryError:
            logging.error(f"OOM on attempt {idx}, trying safer config...")
            torch.cuda.empty_cache()
            try:
                model.use_checkpointing()
            except Exception:
                pass
    raise RuntimeError("All fallback attempts failed due to OOM.")


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    dir_img = Path(args.img_dir)
    dir_mask = Path(args.mask_dir)
    logging.info(f'Using images dir: {dir_img}')
    logging.info(f'Using masks dir: {dir_mask}')
    device = resolve_device(args.device)
    logging.info(f'Using device {device}')
    configure_stable_runtime(args.stable_mode)

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        run_with_oom_fallback(model, device, args)
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)
