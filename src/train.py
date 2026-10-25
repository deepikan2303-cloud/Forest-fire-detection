"""Train a fire / no-fire CNN classifier.

Two-stage transfer learning:
  1. ``--freeze-epochs`` epochs training only the new classification head,
  2. then fine-tune the whole network with a 10x lower backbone learning rate.

Example::

    python -m src.train --arch mobilenet_v3_large --epochs 20
    python -m src.train --arch simple_cnn --epochs 40 --lr 3e-3      # from scratch
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from .dataset import build_dataloaders
from .metrics import binary_metrics, format_metrics
from .model import ARCHS, build_model, count_params, head_module, set_backbone_trainable
from .utils import get_device, set_seed


def run_epoch(model, loader, criterion, device, fire_idx, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, probs, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            probs.append(torch.softmax(logits.float(), dim=1)[:, fire_idx].detach().cpu())
            targets.append(y.cpu())
    probs = torch.cat(probs).numpy()
    is_fire = torch.cat(targets).numpy() == fire_idx
    metrics = binary_metrics(is_fire, probs)
    metrics["loss"] = total_loss / len(probs)
    return metrics


def save_curves(history: list, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "val"):
        axes[0].plot(epochs, [h[split]["loss"] for h in history], label=split)
        axes[1].plot(epochs, [h[split]["f1"] for h in history], label=f"{split} F1")
    axes[1].plot(epochs, [h["val"]["recall"] for h in history], "--", label="val recall")
    axes[0].set_title("Loss")
    axes[1].set_title("Fire-class F1 / recall")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--arch", default="mobilenet_v3_large", choices=ARCHS)
    p.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--freeze-epochs", type=int, default=3, help="head-only epochs before fine-tuning")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--balance", action="store_true", help="class-balanced sampling for imbalanced data")
    p.add_argument("--patience", type=int, default=6, help="early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.workers, args.balance)
    if "fire" not in class_names:
        raise ValueError(f"Expected a 'fire' class folder, found {class_names}")
    fire_idx = class_names.index("fire")
    print(f"Device: {device} | classes: {class_names} | train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)}")

    pretrained = not args.no_pretrained and args.arch != "simple_cnn"
    model = build_model(args.arch, len(class_names), pretrained=pretrained).to(device)
    freeze_epochs = args.freeze_epochs if pretrained else 0
    if freeze_epochs:
        set_backbone_trainable(model, args.arch, False)
    total, trainable = count_params(model)
    print(f"Model: {args.arch} (pretrained={pretrained}) | params={total:,} trainable={trainable:,}")

    # All parameters go in the optimizer up-front; frozen ones simply have no gradient yet.
    head_ids = {id(q) for q in head_module(model, args.arch).parameters()}
    backbone_lr = args.lr * 0.1 if pretrained else args.lr
    optimizer = torch.optim.AdamW([
        {"params": [q for q in model.parameters() if id(q) not in head_ids], "lr": backbone_lr},
        {"params": [q for q in model.parameters() if id(q) in head_ids], "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history, best_key, stale = [], None, 0
    meta = {"arch": args.arch, "num_classes": len(class_names), "class_names": class_names,
            "fire_idx": fire_idx, "img_size": args.img_size, "threshold": 0.5}

    for epoch in range(1, args.epochs + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, args.arch, True)
            print(f"Unfroze backbone; trainable params={count_params(model)[1]:,}")
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, device, fire_idx, optimizer, scaler)
        val_m = run_epoch(model, val_loader, criterion, device, fire_idx)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(f"[{epoch:02d}/{args.epochs}] {time.time() - t0:5.1f}s | "
              f"train loss={train_m['loss']:.4f} f1={train_m['f1']:.4f} | "
              f"val loss={val_m['loss']:.4f} {format_metrics(val_m)}")

        # Model selection on fire-class F1, ties broken by lower validation loss.
        key = (round(val_m["f1"], 4), -val_m["loss"])
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            torch.save({**meta, "epoch": epoch, "val_metrics": val_m, "model_state": model.state_dict()},
                       out_dir / "best_model.pt")
            print(f"  -> saved best model (val f1={val_m['f1']:.4f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    torch.save({**meta, "epoch": epoch, "model_state": model.state_dict()}, out_dir / "last_model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2, default=float)
    save_curves(history, out_dir / "training_curves.png")
    best = max(history, key=lambda h: (round(h["val"]["f1"], 4), -h["val"]["loss"]))
    print(f"\nBest epoch {best['epoch']}: {format_metrics(best['val'])}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}  |  next: python -m src.evaluate")


if __name__ == "__main__":
    main()
