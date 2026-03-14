# pipeline.py — v10
# Pesos: v7 (mejor resultado histórico F1=0.530)
# Todo lo demás: igual a v7 — resize 64x64, augmentation probada
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as tv_transforms
from medmnist import DermaMNIST, INFO
from monai.transforms import (
    Compose, NormalizeIntensity, RandFlip, RandRotate,
    RandZoom, RandGaussianNoise, RandAdjustContrast, Resize,
)

MEAN          = [0.7631, 0.5381, 0.5614]
STD           = [0.1365, 0.1542, 0.1691]

# Pesos v7 — el mejor resultado histórico (F1-test=0.530, AUC=0.9275)
# No tocamos los pesos: el problema no era configuración sino capacidad.
# La skip connection en model.py es la que ahora mueve la aguja.
CLASS_WEIGHTS = [1.20, 1.40, 1.00, 2.00, 1.10, 0.48, 0.89]

CLASS_NAMES   = list(INFO["dermamnist"]["label"].values())
NUM_CLASSES   = len(CLASS_NAMES)

train_transforms = Compose([
    Resize(spatial_size=(64, 64)),
    NormalizeIntensity(subtrahend=MEAN, divisor=STD, channel_wise=True),
    RandFlip(spatial_axis=1, prob=0.5),
    RandFlip(spatial_axis=0, prob=0.5),
    RandRotate(range_x=0.26, prob=0.4, keep_size=True, padding_mode="reflection"),
    RandZoom(min_zoom=0.90, max_zoom=1.10, prob=0.3, keep_size=True, padding_mode="reflect"),
    RandGaussianNoise(prob=0.2, mean=0.0, std=0.02),
    RandAdjustContrast(prob=0.2, gamma=(0.85, 1.25)),
])

val_test_transforms = Compose([
    Resize(spatial_size=(64, 64)),
    NormalizeIntensity(subtrahend=MEAN, divisor=STD, channel_wise=True),
])

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

def get_dataloaders(batch_size=64, num_workers=0):
    train_ds = DermaMNISTDataset("train", transform=train_transforms)
    val_ds   = DermaMNISTDataset("val",   transform=val_test_transforms)
    test_ds  = DermaMNISTDataset("test",  transform=val_test_transforms)

    sample_weights = torch.tensor([CLASS_WEIGHTS[l] for l in train_ds.get_labels()])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader