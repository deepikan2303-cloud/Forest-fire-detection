"""Shared helpers: seeding, device selection, image I/O and checkpoint loading."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import build_model

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(root) -> list:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in IMG_EXTENSIONS else []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def imread(path) -> np.ndarray | None:
    """cv2.imread that also works with non-ASCII paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def imwrite(path, image: np.ndarray, quality: int = 95) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return ok


def load_model(checkpoint_path, device: torch.device):
    """Rebuild a model from a training checkpoint. Returns (model, metadata)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "model_state"}
    return model, meta
