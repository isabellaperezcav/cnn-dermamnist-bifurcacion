# pipeline.py — v14
# BASE: v13 (fusiones correctas del taller + skip connections)
# F1 histórico: v13=0.5951, val_best=0.6156
#
# Cambios de pesos vs v13 [1.20,1.40,1.00,2.00,1.10,0.48,0.89]:
#   actinic[0]:  1.20→1.50  F1=0.44, prec≈rec, clase más rezagada
#   basal[1]:    1.40→1.20  rec=0.66 ya OK, prec=0.49 baja → reducir FP
#   benign[2]:   1.00→1.10  leve boost
#   dermato[3]:  2.00→2.20  rec=0.52 mejoró pero sigue bajo
#   melanoma[4]: 1.10→0.90  prec=0.42 con rec=0.65 → frenar falsos positivos
#   nevi/vascular: sin cambio (✅ estables)
#
# ADVERTENCIA APRENDIDA: nunca bajar melanoma más de -0.20 en un paso
# (en v11 bajamos a 0.70 y rec cayó de 0.71→0.39 — desastre)

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
CLASS_WEIGHTS = [1.50, 1.20, 1.10, 2.20, 0.90, 0.48, 0.89]
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