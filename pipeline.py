# pipeline.py — v2: Augmentation suavizada para imágenes 28x28
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as tv_transforms
from medmnist import DermaMNIST, INFO
from monai.transforms import (
    Compose, NormalizeIntensity, RandFlip, RandRotate,
    RandZoom, RandGaussianNoise, RandAdjustContrast,
)

# ── Constantes ──────────────────────────────────────────
MEAN          = [0.7631, 0.5381, 0.5614]
STD           = [0.1365, 0.1542, 0.1691]

# CAMBIO 1: pesos suavizados con raíz cuadrada
# Original: [4.39, 2.79, 1.30, 12.51, 1.29, 0.21, 10.11]
# v2:       raíz cuadrada → menos agresivo, permite que la clase mayoritaria aprenda
CLASS_WEIGHTS = [2.09, 1.67, 1.14, 3.54, 1.13, 0.46, 3.18]

CLASS_NAMES   = list(INFO["dermamnist"]["label"].values())
NUM_CLASSES   = len(CLASS_NAMES)

# ── Transforms v2: más conservadores para 28x28 ─────────
train_transforms = Compose([
    NormalizeIntensity(subtrahend=MEAN, divisor=STD, channel_wise=True),
    RandFlip(spatial_axis=1, prob=0.5),
    RandFlip(spatial_axis=0, prob=0.5),
    # rotación máx ±15° (antes ±30°), prob 0.4 (antes 0.7)
    RandRotate(range_x=0.26, prob=0.4, keep_size=True, padding_mode="reflection"),
    # zoom conservador 0.9-1.1 (antes 0.85-1.15)
    RandZoom(min_zoom=0.90, max_zoom=1.10, prob=0.3, keep_size=True, padding_mode="reflect"),
    # ruido muy suave std=0.02 (antes 0.05)
    RandGaussianNoise(prob=0.2, mean=0.0, std=0.02),
    # contraste suave gamma=(0.85,1.25) (antes 0.75-1.5)
    RandAdjustContrast(prob=0.2, gamma=(0.85, 1.25)),
])

val_test_transforms = Compose([
    NormalizeIntensity(subtrahend=MEAN, divisor=STD, channel_wise=True),
])

# ── Dataset ─────────────────────────────────────────────
class DermaMNISTDataset(Dataset):
    def __init__(self, split, transform=None):
        self.dataset = DermaMNIST(split=split, transform=tv_transforms.ToTensor(), download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=torch.long).squeeze()
        elif isinstance(label, torch.Tensor):
            label = label.long().squeeze()
        else:
            label = torch.tensor(int(label), dtype=torch.long)
        return img, label

    def get_labels(self):
        labels = []
        for i in range(len(self.dataset)):
            _, lbl = self.dataset[i]
            if isinstance(lbl, np.ndarray):
                labels.append(int(lbl.squeeze()))
            elif isinstance(lbl, torch.Tensor):
                labels.append(int(lbl.squeeze()))
            else:
                labels.append(int(lbl))
        return labels

# ── DataLoaders ─────────────────────────────────────────
def get_dataloaders(batch_size=64, num_workers=0):
    train_ds = DermaMNISTDataset("train", transform=train_transforms)
    val_ds   = DermaMNISTDataset("val",   transform=val_test_transforms)
    test_ds  = DermaMNISTDataset("test",  transform=val_test_transforms)

    sample_weights = torch.tensor([CLASS_WEIGHTS[l] for l in train_ds.get_labels()])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader