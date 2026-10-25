"""Datasets, augmentation pipelines and data loaders.

Expected layout (produced by ``src.prepare_data``)::

    data/processed/
        train/fire/*.jpg   train/nofire/*.jpg
        val/fire/*.jpg     val/nofire/*.jpg
        test/fire/*.jpg    test/nofire/*.jpg
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_RATIO = 1.14  # resize to 256 then centre-crop 224, the standard ImageNet eval recipe


def build_transforms(img_size: int = 224, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            # Moderate crop scale so small fires are rarely cropped out (that would create label noise).
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # Small hue jitter only: colour is a key fire cue and shifting orange toward green would mislabel.
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),  # haze / defocus
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # occlusion (trees, poles)
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * RESIZE_RATIO)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_dataset(data_dir, split: str, img_size: int, train: bool) -> datasets.ImageFolder:
    path = Path(data_dir) / split
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.prepare_data` first.")
    return datasets.ImageFolder(path, transform=build_transforms(img_size, train))


def balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = torch.tensor(dataset.targets)
    counts = torch.bincount(targets, minlength=len(dataset.classes)).float()
    weights = (1.0 / counts.clamp(min=1))[targets]
    return WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)


def build_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=2, balance=False):
    train_ds = make_dataset(data_dir, "train", img_size, train=True)
    val_ds = make_dataset(data_dir, "val", img_size, train=False)
    if val_ds.classes != train_ds.classes:
        raise ValueError(f"Class mismatch: train={train_ds.classes} val={val_ds.classes}")

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    sampler = balanced_sampler(train_ds) if balance else None
    train_loader = DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, train_ds.classes
