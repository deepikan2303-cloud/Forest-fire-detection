"""Deployment-ready inference: single image, image folder, video file or live camera.

    python -m src.predict --source path/to/image.jpg
    python -m src.predict --source path/to/folder --output outputs/predictions.csv
    python -m src.predict --source path/to/video.mp4 --output outputs/annotated.mp4
    python -m src.predict --source 0 --show                      # webcam / USB camera
    python -m src.predict --source rtsp://user:pass@cam/stream   # IP camera

For video, per-frame probabilities are smoothed over a sliding window, and an alert fires
only after the smoothed score stays above the threshold for --min-consecutive checks.
This suppresses one-frame false alarms (sun glare, red vehicles) while still reacting
within a couple of seconds. Alerts are logged to CSV with a snapshot image; hook into
``AlertManager.notify`` to send email/SMS/MQTT.
"""
import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO
from .utils import IMG_EXTENSIONS, get_device, imread, imwrite, list_images, load_model

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class FireDetector:
    """Loads a checkpoint once and scores BGR frames (as returned by OpenCV)."""

    def __init__(self, checkpoint, device="auto", threshold=None):
        self.device = get_device(device)
        self.model, meta = load_model(checkpoint, self.device)
        self.fire_idx = meta["fire_idx"]
        self.img_size = meta["img_size"]
        self.threshold = threshold if threshold is not None else meta.get("threshold", 0.5)
        self.arch = meta["arch"]
        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self._std = np.array(IMAGENET_STD, dtype=np.float32)

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Mirror of the eval transform: resize short side, centre crop, normalise, HWC->CHW."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        short = int(self.img_size * RESIZE_RATIO)
        scale = short / min(h, w)
        rgb = cv2.resize(rgb, (max(short, round(w * scale)), max(short, round(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        top, left = (h - self.img_size) // 2, (w - self.img_size) // 2
        crop = rgb[top:top + self.img_size, left:left + self.img_size].astype(np.float32) / 255.0
        return ((crop - self._mean) / self._std).transpose(2, 0, 1)

    @torch.inference_mode()
    def predict(self, frames: list) -> np.ndarray:
        """Fire probability for each BGR frame."""
        batch = torch.from_numpy(np.stack([self.preprocess(f) for f in frames])).to(self.device)
        return torch.softmax(self.model(batch), dim=1)[:, self.fire_idx].cpu().numpy()


class AlertManager:
    """Debounced alerting with a cooldown, CSV log and snapshot images."""

    def __init__(self, threshold, window=5, min_consecutive=3, cooldown_s=30.0, out_dir="outputs/alerts"):
        self.threshold = threshold
        self.history = deque(maxlen=window)
        self.min_consecutive = min_consecutive
        self.cooldown_s = cooldown_s
        self.out_dir = Path(out_dir)
        self.streak = 0
        self.last_alert_t = -float("inf")
        self.active = False
        self.events = []

    def update(self, prob: float, t: float, frame: np.ndarray, source: str) -> float:
        """Feed one score at stream time ``t`` (seconds). Returns the smoothed score."""
        self.history.append(prob)
        smoothed = float(np.mean(self.history))
        self.streak = self.streak + 1 if smoothed >= self.threshold else 0
        self.active = self.streak >= self.min_consecutive
        if self.active and t - self.last_alert_t >= self.cooldown_s:
            self.last_alert_t = t
            self._fire(smoothed, t, frame, source)
        return smoothed

    def _fire(self, score, t, frame, source):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = self.out_dir / f"alert_{stamp}_{len(self.events):03d}.jpg"
        imwrite(snap, frame)
        event = {"wall_time": datetime.now().isoformat(timespec="seconds"), "stream_time_s": round(t, 2),
                 "score": round(score, 4), "source": source, "snapshot": str(snap)}
        self.events.append(event)
        log = self.out_dir / "alerts.csv"
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(event))
            if new:
                w.writeheader()
            w.writerow(event)
        self.notify(event)

    def notify(self, event: dict) -> None:
        """Override / extend to push notifications (SMTP, Twilio, MQTT, webhook...)."""
        print(f"\a[ALERT] Fire detected at t={event['stream_time_s']}s "
              f"(score={event['score']:.2f}) -> {event['snapshot']}")


def annotate(frame: np.ndarray, prob: float, smoothed: float, threshold: float, alert: bool, fps: float = 0):
    out = frame.copy()
    h, w = out.shape[:2]
    color = (0, 0, 255) if alert else ((0, 165, 255) if smoothed >= threshold else (0, 200, 0))
    label = "FIRE ALERT" if alert else ("possible fire" if smoothed >= threshold else "no fire")
    cv2.rectangle(out, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(out, f"{label}  p={prob:.2f}  avg={smoothed:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    bar_w = int((w - 20) * smoothed)
    cv2.rectangle(out, (10, 38), (10 + bar_w, 42), color, -1)
    tx = int(10 + (w - 20) * threshold)
    cv2.line(out, (tx, 34), (tx, 44), (255, 255, 255), 1)
    if fps:
        cv2.putText(out, f"{fps:.1f} FPS", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if alert:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    return out


def run_images(detector: FireDetector, paths: list, args) -> None:
    rows = []
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        frames = [imread(p) for p in chunk]
        valid = [(p, f) for p, f in zip(chunk, frames) if f is not None]
        if not valid:
            continue
        probs = detector.predict([f for _, f in valid])
        for (path, frame), prob in zip(valid, probs):
            label = "fire" if prob >= detector.threshold else "nofire"
            rows.append({"file": str(path), "fire_prob": round(float(prob), 4), "prediction": label})
            print(f"{label:<7} {prob:6.3f}  {path}")
            if args.save_annotated:
                imwrite(Path(args.save_annotated) / Path(path).name,
                        annotate(frame, prob, prob, detector.threshold, prob >= detector.threshold))
    n_fire = sum(r["prediction"] == "fire" for r in rows)
    print(f"\n{n_fire}/{len(rows)} images flagged as fire (threshold {detector.threshold:.2f})")
    if args.output and len(rows) > 1:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "fire_prob", "prediction"])
            w.writeheader()
            w.writerows(rows)
        print(f"Predictions written to {args.output}")


def run_stream(detector: FireDetector, source: str, args) -> None:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    live = source.isdigit() or "://" in source
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    alerts = AlertManager(detector.threshold, args.window, args.min_consecutive, args.cooldown, args.alert_dir)

    writer = None
    frame_idx, prob, smoothed, fps, t_start = 0, 0.0, 0.0, 0.0, time.time()
    proc_times = deque(maxlen=30)
    print(f"Streaming {source} (fps={src_fps:.1f}, scoring every {args.stride} frame(s)); press q to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (time.time() - t_start) if live else frame_idx / src_fps
            if frame_idx % args.stride == 0:
                t0 = time.time()
                prob = float(detector.predict([frame])[0])
                smoothed = alerts.update(prob, t, frame, source)
                proc_times.append(time.time() - t0)
            fps = 1.0 / np.mean(proc_times) if proc_times else 0.0
            vis = annotate(frame, prob, smoothed, detector.threshold, alerts.active, fps)

            if args.output:
                if writer is None:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(vis)
            if args.show:
                cv2.imshow("Forest Fire Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\nProcessed {frame_idx} frames, model throughput ~{fps:.1f} FPS on {detector.device}")
    print(f"{len(alerts.events)} alert(s) raised" + (f"; log: {alerts.out_dir / 'alerts.csv'}" if alerts.events else ""))
    if args.output:
        print(f"Annotated video: {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="image, folder, video file, camera index or stream URL")
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--threshold", type=float, help="override the checkpoint's threshold")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", help="CSV for folders, annotated .mp4 for video")
    p.add_argument("--save-annotated", help="folder to save annotated images (image mode)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--show", action="store_true", help="display a live window (video mode)")
    p.add_argument("--stride", type=int, default=5, help="score every N-th frame (video mode)")
    p.add_argument("--window", type=int, default=5, help="smoothing window, in scored frames")
    p.add_argument("--min-consecutive", type=int, default=3, help="smoothed hits needed before alerting")
    p.add_argument("--cooldown", type=float, default=30.0, help="seconds between repeated alerts")
    p.add_argument("--alert-dir", default="outputs/alerts")
    args = p.parse_args()

    detector = FireDetector(args.checkpoint, args.device, args.threshold)
    print(f"Loaded {detector.arch} on {detector.device}, threshold={detector.threshold:.2f}")

    src = args.source
    path = Path(src)
    if path.is_dir():
        run_images(detector, list_images(path), args)
    elif path.suffix.lower() in IMG_EXTENSIONS:
        run_images(detector, [path], args)
    elif src.isdigit() or "://" in src or path.suffix.lower() in VIDEO_EXTENSIONS:
        run_stream(detector, src, args)
    else:
        p.error(f"Unrecognised source: {src}")


if __name__ == "__main__":
    main()
