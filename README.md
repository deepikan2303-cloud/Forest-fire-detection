# Forest Fire Detection using CNN

A complete PyTorch pipeline that learns to tell **fire** from **no fire** in images and video frames, built for early wildfire alerts:

**dataset curation → augmentation → transfer-learning CNN → evaluation and threshold tuning → real-time video inference with alerts → TorchScript/ONNX export**

```
Forest Fire Detection/
├── src/
│   ├── prepare_data.py   # curation: validate, dedupe (perceptual hash), resize, stratified split
│   ├── dataset.py        # augmentation pipelines, DataLoaders, class-balanced sampling
│   ├── model.py          # MobileNetV3 / ResNet18 / EfficientNet-B0 (TorchVision) + SimpleCNN from scratch
│   ├── train.py          # two-stage fine-tuning, cosine LR, AMP, early stopping, curves
│   ├── metrics.py        # precision / recall / F1 / ROC-AUC, threshold search (NumPy only)
│   ├── evaluate.py       # test metrics, confusion matrix, misclassified list, threshold tuning
│   ├── predict.py        # image / folder / video / webcam / RTSP inference + debounced alerts
│   └── export.py         # TorchScript + ONNX export with parity checks
├── scripts/make_demo_data.py   # synthetic data + demo video for smoke testing only
├── data/      # raw/ (your images), processed/ (generated splits)
├── models/    # checkpoints, history.json, training_curves.png, export/
└── outputs/   # evaluation reports, predictions, alerts/
```

## 1. Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
# CPU build (works everywhere):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# or, with an NVIDIA GPU, pick the CUDA build from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

Run every command from the project root (`python -m src.<module>`). Add `--help` to any command to see all its options.

## 2. Get a dataset

Download one or more public wildfire datasets. Merging several improves robustness.

| Dataset | Where | Notes |
|---|---|---|
| The Wildfire Dataset | Kaggle `elmadafri/the-wildfire-dataset` | ~2.7k forest images, fire / nofire, includes hard negatives (fog, sunsets) |
| FIRE Dataset | Kaggle `phylake1337/fire-dataset` | ~1k outdoor images (`fire_images`, `non_fire_images`) |
| FLAME | IEEE DataPort ("FLAME dataset") | drone footage frames of prescribed burns, fire / no fire |
| Wildfire Prediction | Kaggle `abdelghaniaaba/wildfire-prediction-dataset` | **satellite** tiles. Use only if you deploy on satellite imagery |

You can download from Kaggle with the CLI (`pip install kaggle`, then put your API token in `%USERPROFILE%\.kaggle\kaggle.json`):

```powershell
kaggle datasets download -d elmadafri/the-wildfire-dataset -p data/downloads/wildfire --unzip
kaggle datasets download -d phylake1337/fire-dataset       -p data/downloads/firedataset --unzip
```

## 3. Curate and split

`prepare_data` accepts any number of folders per class and searches them recursively, so you can point it straight at the downloaded layouts. Check the actual folder names after unzipping:

```powershell
python -m src.prepare_data `
  --fire   data/downloads/wildfire/<...>/fire   data/downloads/firedataset/fire_dataset/fire_images `
  --nofire data/downloads/wildfire/<...>/nofire data/downloads/firedataset/fire_dataset/non_fire_images
```

What it does:
- removes corrupt or unreadable files and images under 64 px
- removes exact and **near-duplicates** (64-bit dHash, Hamming distance ≤ 4). Wildfire datasets are full of consecutive video frames. Without this step, near-identical frames end up in both train and test, and test accuracy looks far better than it really is.
- downscales large images to a 640 px longest side
- makes a stratified, seeded 70/15/15 split into `data/processed/{train,val,test}/{fire,nofire}`
- writes `manifest.csv`, which maps every file back to its source

## 4. Train

```powershell
python -m src.train --arch mobilenet_v3_large --epochs 20          # recommended default
python -m src.train --arch efficientnet_b0 --epochs 25 --balance    # stronger, slower; balanced sampling
python -m src.train --arch simple_cnn --epochs 40 --lr 3e-3         # CNN from scratch (baseline)
```

How training works:
- **Transfer learning in two stages.** For the first `--freeze-epochs` (default 3) epochs only the new head is trained. Then the whole network is fine-tuned, with the backbone at a 10× lower learning rate so the pretrained features aren't destroyed early on.
- **Augmentation** (`src/dataset.py`):
  - random resized crop (scale ≥ 0.6, so a small fire is rarely cropped out of a "fire" image)
  - flips and ±10° rotation
  - brightness/contrast/saturation jitter (lighting and time of day)
  - **very small hue jitter**, because colour is a key fire cue and turning orange into green would create label noise
  - Gaussian blur (haze and smoke)
  - random erasing (occlusion by branches)
- **Regularisation:** AdamW, label smoothing 0.1, cosine LR schedule, early stopping on **fire-class F1**.
- **Mixed precision** runs automatically on CUDA.

Outputs: `models/best_model.pt`, `models/last_model.pt`, `models/history.json`, `models/training_curves.png`.

## 5. Evaluate and pick the alert threshold

```powershell
python -m src.evaluate                                     # test metrics at threshold 0.5
python -m src.evaluate --target-recall 0.97 --save-threshold
```

In early warning, a **missed fire (false negative) costs far more than a false alarm**. `--target-recall` picks the highest threshold that still catches the requested share of fires. The threshold is tuned on the **validation** split (the test split stays unbiased), and `--save-threshold` stores it in the checkpoint. Inference uses it automatically from then on.

Outputs in `outputs/`:
- `eval_test.json`
- `confusion_matrix_test.png`
- `misclassified_test.csv`, sorted by how confidently wrong the model was. **Look at these images.** They show mislabelled data and the kinds of hard negatives you need to add more of.

## 6. Inference / deployment

```powershell
python -m src.predict --source some_image.jpg
python -m src.predict --source data/processed/test --output outputs/predictions.csv
python -m src.predict --source drone_clip.mp4 --output outputs/annotated.mp4
python -m src.predict --source 0 --show                       # webcam (press q to quit)
python -m src.predict --source rtsp://user:pass@ip/stream     # IP / CCTV camera
```

How the video pipeline avoids false alarms:
1. It scores every `--stride` frames (default 5), which saves compute on edge devices.
2. It averages the probability over the last `--window` scores (default 5).
3. It raises an alert only when the smoothed score stays above the threshold for `--min-consecutive` checks (default 3). This filters single-frame false positives such as sun glare or a red truck.
4. A `--cooldown` (default 30 s) stops the same fire from triggering repeated alerts.

Each alert is printed and appended to `outputs/alerts/alerts.csv`, and a snapshot JPEG is saved. To send email, SMS, MQTT or webhook notifications, extend `AlertManager.notify()` in [src/predict.py](src/predict.py).

The `FireDetector` class can also be used from your own code:

```python
from src.predict import FireDetector
det = FireDetector("models/best_model.pt")
prob = det.predict([cv2.imread("frame.jpg")])[0]
```

### Export

```powershell
python -m src.export        # -> models/export/*.torchscript.pt, *.onnx, *.json
```

You can run the ONNX model with ONNX Runtime, OpenCV DNN, OpenVINO or TensorRT, for example on a Jetson or Raspberry Pi. The JSON file holds the preprocessing constants and the alert threshold.

## 7. Smoke test without downloading anything

```powershell
python scripts/make_demo_data.py --per-class 300 --video
python -m src.prepare_data
python -m src.train --arch mobilenet_v3_small --epochs 5 --freeze-epochs 1
python -m src.evaluate --save-threshold
python -m src.predict --source data/demo_video.mp4 --output outputs/demo_annotated.mp4
python -m src.export
```

The synthetic images are cartoons. They only prove that the pipeline runs end to end, so **retrain on a real dataset** before drawing any conclusions.

## Model choice

| Arch | Params | CPU speed | When to use |
|---|---|---|---|
| `mobilenet_v3_small` | 1.5M | fastest | Raspberry Pi, very low-power cameras |
| `mobilenet_v3_large` | 4.2M | fast | **default**: good accuracy/speed trade-off for edge use |
| `efficientnet_b0` | 4.0M | medium | best accuracy if you have a GPU for training |
| `resnet18` | 11.2M | medium | simple, robust baseline |
| `simple_cnn` | 1.2M | fast | from-scratch CNN, to compare against transfer learning |

## Tips for real-world accuracy

- **Hard negatives matter most.** Add sunsets, autumn foliage, fog and low cloud, red or orange objects, street lights at night, and lens flare to the `nofire` class.
- Add **smoke-only** images (early-stage fires) to the `fire` class if you want earlier detection.
- Match your deployment viewpoint: tower camera, drone or ground. A model trained only on ground photos transfers poorly to aerial views.
- Classification says *whether* there is fire, not *where*. To get bounding boxes, move to a detector such as YOLO trained on the D-Fire dataset. You can reuse this curation and alert pipeline for it.
