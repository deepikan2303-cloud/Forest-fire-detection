"""Dataset curation: validate, de-duplicate, resize and split raw images into train/val/test.

Point it at any number of source folders per class (searched recursively), so several
public datasets with different layouts can be merged in one go::

    python -m src.prepare_data \
        --fire    data/raw/fire  downloads/wildfire_dataset/train/fire \
        --nofire  data/raw/nofire downloads/wildfire_dataset/train/nofire

Curation steps
  1. Drop unreadable / corrupt files.
  2. Drop images smaller than --min-size pixels on the short side.
  3. Drop exact and near-duplicates using a 64-bit perceptual difference hash (dHash).
     Public wildfire sets contain many consecutive video frames; leaving them in lets
     near-identical images land in both train and test, inflating test accuracy.
  4. Downscale large images (saves disk and speeds up loading).
  5. Stratified, seeded split into train / val / test, plus a manifest.csv for traceability.
"""
import argparse
import csv
import os
import random
import shutil
import stat
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .utils import imread, imwrite, list_images

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(image: np.ndarray) -> np.uint64:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


class DuplicateIndex:
    """Stores hashes and answers "is there a hash within `max_dist` bits?"."""

    def __init__(self, max_dist: int):
        self.max_dist = max_dist
        self.hashes = np.empty(0, dtype=np.uint64)
        self._exact = set()

    def seen(self, h: np.uint64) -> bool:
        if int(h) in self._exact:
            return True
        if self.max_dist > 0 and len(self.hashes):
            xor = np.bitwise_xor(self.hashes, h)
            dist = _POPCOUNT[xor.view(np.uint8)].reshape(-1, 8).sum(axis=1)
            if dist.min() <= self.max_dist:
                return True
        return False

    def add(self, h: np.uint64) -> None:
        self._exact.add(int(h))
        self.hashes = np.append(self.hashes, h)


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def curate(sources: dict, args) -> tuple:
    """Return {label: [(src_path, image), ...]} of accepted images and rejection stats."""
    index = DuplicateIndex(0 if args.no_dedupe else args.dedupe_threshold)
    kept = {label: [] for label in sources}
    stats = Counter()
    for label, dirs in sources.items():
        for d in dirs:
            files = list_images(d)
            if not files:
                print(f"  [warn] no images found in {d}")
            for path in files:
                img = imread(path)
                if img is None:
                    stats["corrupt"] += 1
                    continue
                if min(img.shape[:2]) < args.min_size:
                    stats["too_small"] += 1
                    continue
                if not args.no_dedupe:
                    h = dhash(img)
                    if index.seen(h):
                        stats["duplicate"] += 1
                        continue
                    index.add(h)
                kept[label].append((path, resize_max_side(img, args.max_side)))
                stats[f"kept_{label}"] += 1
    return kept, stats


def split_and_write(kept: dict, out: Path, ratios, seed: int) -> list:
    rng = random.Random(seed)
    manifest = []
    for label, items in kept.items():
        rng.shuffle(items)
        n = len(items)
        n_train, n_val = round(n * ratios[0]), round(n * ratios[1])
        splits = {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}
        for split, subset in splits.items():
            (out / split / label).mkdir(parents=True, exist_ok=True)
            for i, (src, img) in enumerate(subset):
                dst = out / split / label / f"{label}_{i:05d}.jpg"
                imwrite(dst, img)
                manifest.append({"split": split, "label": label, "file": dst.relative_to(out).as_posix(),
                                 "source": str(src)})
    return manifest


def _clear_readonly(func, p, _exc):
    os.chmod(p, stat.S_IWRITE)  # OneDrive marks synced folders read-only, which blocks rmdir
    func(p)


def remove_tree(path: Path, attempts: int = 5) -> None:
    """rmtree that clears read-only flags and retries: OneDrive / Explorer briefly lock folders on Windows."""
    for i in range(attempts):
        try:
            shutil.rmtree(path, onerror=_clear_readonly)
            return
        except PermissionError:
            if i == attempts - 1:
                raise PermissionError(f"Could not delete {path}. Close any Explorer window or "
                                      f"image viewer showing it (or pause OneDrive sync) and retry.")
            time.sleep(1.0 + i)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fire", nargs="+", default=["data/raw/fire"], help="folders with fire images")
    p.add_argument("--nofire", nargs="+", default=["data/raw/nofire"], help="folders with non-fire images")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--min-size", type=int, default=64)
    p.add_argument("--max-side", type=int, default=640, help="downscale longer side to this (0 = keep)")
    p.add_argument("--dedupe-threshold", type=int, default=4, help="max dHash Hamming distance for near-dupes")
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if abs(sum(args.split) - 1.0) > 1e-6:
        p.error("--split ratios must sum to 1")
    out = Path(args.out)
    if out.exists():
        print(f"Removing existing {out}")
        remove_tree(out)

    print("Scanning and curating images ...")
    kept, stats = curate({"fire": args.fire, "nofire": args.nofire}, args)
    manifest = split_and_write(kept, out, args.split, args.seed)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "file", "source"])
        writer.writeheader()
        writer.writerows(manifest)

    print("\nRejected: " + ", ".join(f"{k}={stats[k]}" for k in ("corrupt", "too_small", "duplicate")))
    counts = Counter((m["split"], m["label"]) for m in manifest)
    print(f"\n{'split':<8}{'fire':>8}{'nofire':>8}")
    for split in ("train", "val", "test"):
        print(f"{split:<8}{counts[(split, 'fire')]:>8}{counts[(split, 'nofire')]:>8}")
    n_fire, n_nofire = stats["kept_fire"], stats["kept_nofire"]
    if min(n_fire, n_nofire) and max(n_fire, n_nofire) / min(n_fire, n_nofire) > 1.5:
        print("\n[note] classes are imbalanced; consider training with --balance")
    print(f"\nWrote {len(manifest)} images to {out}")


if __name__ == "__main__":
    main()
