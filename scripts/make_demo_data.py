"""Generate a SYNTHETIC forest dataset (and an optional demo video) to smoke-test the pipeline.

These images are cartoons, not real wildfire data: a model trained on them proves the code
runs end to end but is useless in the field. Use a real dataset (see README) for training.

    python scripts/make_demo_data.py --per-class 300 --video
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import imwrite  # noqa: E402


def forest_scene(rng: np.random.Generator, size: int = 256, sunset: bool = False) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    horizon = int(size * rng.uniform(0.3, 0.5))
    top = np.array([200, 120, 60]) if not sunset else np.array([80, 120, 230])   # BGR: blue sky / orange sunset
    bottom = np.array([235, 200, 160]) if not sunset else np.array([120, 190, 255])
    for y in range(horizon):
        a = y / max(horizon - 1, 1)
        img[y] = (top * (1 - a) + bottom * a).astype(np.uint8)
    ground = np.array([30, rng.integers(70, 110), 40]) if not sunset else np.array([20, 50, 30])
    img[horizon:] = ground
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    for _ in range(rng.integers(8, 20)):  # trees
        x = int(rng.integers(0, size))
        base = int(rng.integers(horizon, size))
        hgt = int(rng.integers(30, 90))
        wid = hgt // 3
        green = (int(rng.integers(10, 40)), int(rng.integers(60, 120)), int(rng.integers(10, 40)))
        cv2.rectangle(img, (x - 2, base - 10), (x + 2, base), (20, 40, 70), -1)
        pts = np.array([[x, base - 10 - hgt], [x - wid, base - 10], [x + wid, base - 10]], np.int32)
        cv2.fillPoly(img, [pts], green)
    return img


def add_fire(img: np.ndarray, rng: np.random.Generator, intensity: float = 1.0) -> np.ndarray:
    size = img.shape[0]
    overlay = np.zeros_like(img, dtype=np.float32)
    smoke = img.astype(np.float32)
    for _ in range(rng.integers(1, 4)):
        cx, cy = int(rng.integers(20, size - 20)), int(rng.integers(size // 2, size - 10))
        r = int(rng.integers(12, 40) * intensity) + 4
        for _ in range(3):  # smoke plume
            sx, sy = cx + int(rng.integers(-15, 15)), cy - int(rng.integers(r, 3 * r))
            cv2.circle(smoke, (sx, sy), int(r * rng.uniform(1.0, 2.0)), (150, 150, 150), -1)
        for color, scale in (((0, 40, 200), 1.0), ((0, 120, 255), 0.7), ((80, 230, 255), 0.4)):
            for _ in range(4):  # flame tongues: red -> orange -> yellow core
                ax = (max(int(r * scale * rng.uniform(0.4, 0.8)), 2), max(int(r * scale * rng.uniform(1.0, 1.8)), 2))
                cen = (cx + int(rng.integers(-r // 2, r // 2 + 1)), cy - ax[1] // 2)
                cv2.ellipse(overlay, cen, ax, 0, 0, 360, color, -1)
    smoke = cv2.GaussianBlur(smoke, (31, 31), 0)
    out = cv2.addWeighted(img.astype(np.float32), 0.45, smoke, 0.55, 0)
    mask = (overlay.sum(axis=2, keepdims=True) > 0).astype(np.float32)
    glow = cv2.GaussianBlur(overlay, (15, 15), 0)
    out = out * (1 - mask) + overlay * mask
    out = np.clip(out + 0.6 * glow, 0, 255)
    return out.astype(np.uint8)


def make_video(path: Path, rng: np.random.Generator, seconds: int = 12, fps: int = 15) -> None:
    """Forest scene with a fire that starts halfway through, with a slow camera pan."""
    base = cv2.resize(forest_scene(rng, 256), (640, 480))
    fire = add_fire(forest_scene(np.random.default_rng(0), 256), np.random.default_rng(1))
    fire = cv2.resize(fire, (640, 480))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    n = seconds * fps
    for i in range(n):
        a = np.clip((i - n / 2) / (fps * 1.5), 0, 1)  # fade fire in over 1.5 s after the midpoint
        frame = cv2.addWeighted(base, 1 - a, fire, a, 0)
        frame = np.roll(frame, int(3 * np.sin(i / 10)), axis=1)
        noise = rng.normal(0, 3, frame.shape)
        writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--video", action="store_true", help="also write data/demo_video.mp4")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for i in range(args.per_class):
        # ~20% sunset scenes in BOTH classes so "orange = fire" is not a shortcut.
        imwrite(out / "nofire" / f"synthetic_nofire_{i:04d}.jpg", forest_scene(rng, sunset=rng.random() < 0.2))
        scene = forest_scene(rng, sunset=rng.random() < 0.2)
        imwrite(out / "fire" / f"synthetic_fire_{i:04d}.jpg", add_fire(scene, rng, rng.uniform(0.5, 1.3)))
    print(f"Wrote {2 * args.per_class} synthetic images to {out}/fire and {out}/nofire")
    if args.video:
        path = out.parent / "demo_video.mp4"
        make_video(path, rng)
        print(f"Wrote demo video to {path} (fire appears at ~6 s)")


if __name__ == "__main__":
    main()
