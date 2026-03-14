# model.py — v2 con skip connections residuales
# Las 4 FUSIONES MATEMÁTICAS del taller se mantienen INTACTAS.
# Único cambio: se añade una proyección residual Conv1x1 DESPUÉS de la fusión.
#
# Antes:  out = pool(fusion(branch1(x), branch2(x)))
# Ahora:  out = pool(fusion(branch1(x), branch2(x)) + proj(x))
#
# La proyección Conv1x1 es necesaria porque in_channels ≠ out_channels
# en todos los bloques (3→50, 50→100, 100→75, 75→25).
# Añade solo 14,525 parámetros extra (+2.9% sobre 495,032 base).

import torch
import torch.nn as nn


# ── Módulos de Fusión (SIN CAMBIOS) ─────────────────────────────────────────

class EuclideanFusion(nn.Module):
    """Bloque 1: sqrt(X1^2 + X2^2 + eps) — Realce de bordes"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x1, x2):
        return torch.sqrt(x1 ** 2 + x2 ** 2 + self.eps)


class HadamardMaxFusion(nn.Module):
    """Bloque 2: (X1 * X2) + max(X1, X2) — Potenciación de texturas"""
    def forward(self, x1, x2):
        return (x1 * x2) + torch.maximum(x1, x2)


class PseudoRadialFusion(nn.Module):
    """Bloque 3: 2*X1*X2 / (X1^2 + X2^2 + eps) — Consolidación de formas"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x1, x2):
        return (2 * x1 * x2) / (x1 ** 2 + x2 ** 2 + self.eps)


class CrossAttentionFusion(nn.Module):
    """Bloque 4: sigmoid(X1) * X2 — Atención cruzada semántica"""
    def forward(self, x1, x2):
        return torch.sigmoid(x1) * x2


# ── Bloque Genérico CON Skip Connection ─────────────────────────────────────

class MultiscaleBlock(nn.Module):
    """
    Bloque multiescala con skip connection residual.

    Flujo:
        x1     = branch1(x)          # Conv3x3 + BN + ReLU
        x2     = branch2(x)          # Conv5x5 + BN + ReLU
        fused  = fusion(x1, x2)      # fusión matemática específica (sin cambios)
        skip   = proj(x)             # Conv1x1 para igualar canales
        out    = pool(fused + skip)   # suma residual + reducción espacial

    La skip connection permite que los gradientes fluyan directamente
    hacia atrás, mejorando el aprendizaje de clases con pocas muestras.
    """
    def __init__(self, in_channels, out_channels, fusion, pool=True):
        super().__init__()

        # Rama X1: kernel 3×3 (sin cambios)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Rama X2: kernel 5×5 (sin cambios)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = fusion

        # Proyección residual: Conv1x1 sin activación
        # Necesaria porque in_channels ≠ out_channels en todos los bloques
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        x1    = self.branch1(x)
        x2    = self.branch2(x)
        fused = self.fusion(x1, x2)   # fusión matemática (intacta)
        skip  = self.proj(x)          # proyección residual
        return self.pool(fused + skip) # suma + pool


# ── Red Completa (sin cambios en estructura) ─────────────────────────────────

class DermaCNN(nn.Module):
    """
    CNN Multiescala para DermaMNIST.
    Canales: 3 → 50 → 100 → 75 → 25 → 7
    Las fusiones matemáticas de los 4 bloques son idénticas al taller.
    """
    def __init__(self, num_classes=7, dropout=0.25):
        super().__init__()
        self.block1 = MultiscaleBlock(3,   50,  EuclideanFusion(),      pool=True)
        self.block2 = MultiscaleBlock(50,  100, HadamardMaxFusion(),    pool=True)
        self.block3 = MultiscaleBlock(100, 75,  PseudoRadialFusion(),   pool=True)
        self.block4 = MultiscaleBlock(75,  25,  CrossAttentionFusion(), pool=False)
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