"""Evaluate a checkpoint on the held-out test split and choose the alert threshold.

    python -m src.evaluate                                 # metrics at the stored threshold
    python -m src.evaluate --target-recall 0.97 --save-threshold

Outputs (in --out-dir): eval_<split>.json, confusion_matrix_<split>.png, misclassified_<split>.csv.
Review the misclassified list: it is the fastest way to find label errors in the dataset.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import make_dataset
from .metrics import best_threshold, binary_metrics, format_metrics
from .utils import get_device, load_model


@torch.inference_mode()
def predict_dataset(model, loader, device, fire_idx) -> np.ndarray:
    probs = [torch.softmax(model(x.to(device)), dim=1)[:, fire_idx].cpu() for x, _ in loader]
    return torch.cat(probs).numpy()


def save_confusion_matrix(m: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    labels = ["fire", "nofire"]
    ax.set_xticks([0, 1], [f"pred {l}" for l in labels])
    ax.set_yticks([0, 1], [f"true {l}" for l in labels])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title(f"threshold={m['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, help="override the checkpoint's threshold")
    p.add_argument("--target-recall", type=float, help="tune threshold on VAL to reach this fire recall")
    p.add_argument("--save-threshold", action="store_true", help="write the tuned threshold into the checkpoint")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    device = get_device(args.device)
    model, meta = load_model(args.checkpoint, device)
    fire_idx, img_size = meta["fire_idx"], meta["img_size"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold if args.threshold is not None else meta.get("threshold", 0.5)
    if args.target_recall is not None or args.save_threshold:
        # Tune on validation data, never on the test split, to keep the test estimate unbiased.
        val_ds = make_dataset(args.data_dir, "val", img_size, train=False)
        val_probs = predict_dataset(model, DataLoader(val_ds, batch_size=args.batch_size), device, fire_idx)
        val_true = np.array(val_ds.targets) == fire_idx
        threshold = best_threshold(val_true, val_probs, args.target_recall)
        goal = f"recall>={args.target_recall}" if args.target_recall else "max F1"
        print(f"Tuned threshold on val ({goal}): {threshold:.2f}")

    ds = make_dataset(args.data_dir, args.split, img_size, train=False)
    if ds.classes != meta["class_names"]:
        raise ValueError(f"Class mismatch: data={ds.classes} checkpoint={meta['class_names']}")
    probs = predict_dataset(model, DataLoader(ds, batch_size=args.batch_size), device, fire_idx)
    y_true = np.array(ds.targets) == fire_idx

    m = binary_metrics(y_true, probs, threshold)
    print(f"\n{args.split} set ({len(ds)} images) @ threshold {threshold:.2f}")
    print("  " + format_metrics(m))
    print(f"  confusion: TP={m['tp']} FN={m['fn']} (missed fires) FP={m['fp']} (false alarms) TN={m['tn']}")

    with open(out_dir / f"eval_{args.split}.json", "w") as f:
        json.dump(m, f, indent=2)
    save_confusion_matrix(m, out_dir / f"confusion_matrix_{args.split}.png")
    wrong = [(path, "fire" if t else "nofire", float(pr)) for (path, _), t, pr in zip(ds.samples, y_true, probs)
             if (pr >= threshold) != t]
    with open(out_dir / f"misclassified_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "fire_prob"])
        w.writerows(sorted(wrong, key=lambda r: -abs(r[2] - threshold)))
    print(f"  {len(wrong)} misclassified -> {out_dir / f'misclassified_{args.split}.csv'}")

    if args.save_threshold:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt["threshold"] = threshold
        torch.save(ckpt, args.checkpoint)
        print(f"Saved threshold {threshold:.2f} into {args.checkpoint}")


if __name__ == "__main__":
    main()
