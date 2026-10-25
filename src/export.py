"""Export a trained checkpoint for deployment.

    python -m src.export --format torchscript onnx

TorchScript runs with plain libtorch (C++ / mobile); ONNX runs with ONNX Runtime,
OpenCV DNN, TensorRT or OpenVINO on edge devices (Jetson, Raspberry Pi). Both exported
models take a float32 batch of shape (N, 3, img_size, img_size), ImageNet-normalised
RGB, and return softmax probabilities. A metadata JSON is written next to them.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .utils import load_model


class WithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--format", nargs="+", default=["torchscript", "onnx"], choices=["torchscript", "onnx"])
    p.add_argument("--out-dir", default="models/export")
    args = p.parse_args()
    # TorchScript / legacy ONNX exporters are deprecated but remain the most portable formats for edge runtimes.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    device = torch.device("cpu")
    model, meta = load_model(args.checkpoint, device)
    wrapped = WithSoftmax(model).eval()
    size = meta["img_size"]
    dummy = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        reference = wrapped(dummy).numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fire_{meta['arch']}"

    if "torchscript" in args.format:
        path = out_dir / f"{stem}.torchscript.pt"
        with torch.inference_mode():
            traced = torch.jit.trace(wrapped, dummy)
        traced.save(str(path))
        err = np.abs(torch.jit.load(str(path))(dummy).detach().numpy() - reference).max()
        print(f"TorchScript -> {path}  (max abs diff {err:.2e})")

    if "onnx" in args.format:
        path = out_dir / f"{stem}.onnx"
        try:
            torch.onnx.export(wrapped, dummy, str(path), input_names=["input"], output_names=["probs"],
                              dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17,
                              dynamo=False)
            print(f"ONNX        -> {path}")
        except Exception as exc:
            print(f"ONNX export failed ({exc}). Install with: pip install onnx")
        else:
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                err = np.abs(sess.run(None, {"input": dummy.numpy()})[0] - reference).max()
                print(f"  onnxruntime check: max abs diff {err:.2e}")
            except ImportError:
                print("  (pip install onnxruntime to verify the ONNX model)")

    info = {k: meta[k] for k in ("arch", "class_names", "fire_idx", "img_size", "threshold")}
    info.update(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], input="RGB float32 NCHW, /255 then normalise")
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Metadata    -> {out_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
