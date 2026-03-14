# model.py — v3: fusiones EXACTAS del taller + skip connections residuales
#
# CORRECCIONES respecto a v2:
#   Bloque 3: era  2*X1*X2/(X1²+X2²+ε)  (media armónica — INCORRECTA)
#             ahora exp(-(X1-X2)²/2) ⊙ (X1+X2)  ← fórmula del taller ✅
#
#   Bloque 4: era  σ(X1) ⊙ X2  (unidireccional — INCOMPLETA)
#             ahora σ(X1) ⊙ X2 + σ(X2) ⊙ X1    ← fórmula del taller ✅
#
# Bloques 1 y 2: correctos desde el inicio, sin cambios.
# Skip connections residuales: se mantienen (mejoraron F1 0.53→0.58 en v10).

import torch
import torch.nn as nn


# ── Fusiones exactas del taller ──────────────────────────────────────────────

class EuclideanFusion(nn.Module):
    """Bloque 1: F1 = sqrt(X1² + X2² + ε)"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x1, x2):
        return torch.sqrt(x1 ** 2 + x2 ** 2 + self.eps)


class HadamardMaxFusion(nn.Module):
    """Bloque 2: F2 = (X1 ⊙ X2) + max(X1, X2)"""
    def forward(self, x1, x2):
        return (x1 * x2) + torch.maximum(x1, x2)


class GaussianSimilarityFusion(nn.Module):
    """
    Bloque 3: F3 = exp(-(X1-X2)²/2) ⊙ (X1+X2)
    Gate de similitud gaussiana: amplifica donde X1≈X2, suprime diferencias.
    """
    def forward(self, x1, x2):
        gate = torch.exp(-((x1 - x2) ** 2) / 2)
        return gate * (x1 + x2)


class BilateralAttentionFusion(nn.Module):
    """
    Bloque 4: F4 = σ(X1) ⊙ X2 + σ(X2) ⊙ X1
    Atención cruzada BILATERAL — cada rama atiende a la otra.
    Grad-CAM se calcula sobre este bloque.
    """
    def forward(self, x1, x2):
        return torch.sigmoid(x1) * x2 + torch.sigmoid(x2) * x1


# ── Bloque genérico con skip connection ──────────────────────────────────────

class MultiscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fusion, pool=True):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = fusion
        self.proj   = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        fused = self.fusion(self.branch1(x), self.branch2(x))
        return self.pool(fused + self.proj(x))


# ── Red completa ──────────────────────────────────────────────────────────────

class DermaCNN(nn.Module):
    """
    CNN Multiescala — DermaMNIST.
    Canales: 3 → 50 → 100 → 75 → 25 → 7

    Bloque 1: EuclideanFusion           sqrt(X1²+X2²+ε)
    Bloque 2: HadamardMaxFusion         (X1⊙X2) + max(X1,X2)
    Bloque 3: GaussianSimilarityFusion  exp(-(X1-X2)²/2) ⊙ (X1+X2)
    Bloque 4: BilateralAttentionFusion  σ(X1)⊙X2 + σ(X2)⊙X1
    """
    def __init__(self, num_classes=7, dropout=0.25):
        super().__init__()
        self.block1 = MultiscaleBlock(3,   50,  EuclideanFusion(),          pool=True)
        self.block2 = MultiscaleBlock(50,  100, HadamardMaxFusion(),        pool=True)
        self.block3 = MultiscaleBlock(100, 75,  GaussianSimilarityFusion(), pool=True)
        self.block4 = MultiscaleBlock(75,  25,  BilateralAttentionFusion(), pool=False)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(25, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)