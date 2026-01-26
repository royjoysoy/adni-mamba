"""
File: models_rs/dsa_3dcnn_01_24_2026.py
Date: 01_24_2026
Purpose:
  - Paper-style DSA-3D-CNN (Deep Supervision on FC layers) for ADNI 3-class classification (CN/MCI/AD)
  - Input tensor shape: (B, 1, 96, 96, 96) skull-stripped T1
  - Output:
      - logits_final: (B, 3)
      - logits_ds1:   (B, 3)  [deep supervision head from FC1]
      - logits_ds2:   (B, 3)  [deep supervision head from FC2]
  - Loss:
      total = w1*CE(logits_ds1,y) + w2*CE(logits_ds2,y) + w3*CE(logits_final,y)

How to use (typical):
  from models_rs.dsa_3dcnn_01_24_2026 import DSA3DCNN, deep_supervision_loss

Notes:
  - This module is standalone. Your training script should:
      1) create model = DSA3DCNN(in_ch=1, n_classes=3)
      2) forward with return_all_logits=True
      3) compute deep_supervision_loss(...)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSA3DCNN(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 3, dropout: float = 0.4):
        super().__init__()

        # Encoder: 96 -> 48 -> 24 -> 12
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.flat_dim = 128 * 12 * 12 * 12

        # FC stack
        self.fc1 = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc3 = nn.Linear(256, n_classes)

        # Deep supervision heads
        self.ds1 = nn.Linear(512, n_classes)
        self.ds2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor, return_all_logits: bool = False):
        """
        Input:
          x: (B, 1, 96, 96, 96)
        Output:
          if return_all_logits=False:
            logits_final: (B, n_classes)
          else:
            dict with logits_final/logits_ds1/logits_ds2
        """
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = x.reshape(x.size(0), -1)

        h1 = self.fc1(x)          # (B, 512)
        h2 = self.fc2(h1)         # (B, 256)
        logits_final = self.fc3(h2)

        if not return_all_logits:
            return logits_final

        return {
            "logits_final": logits_final,
            "logits_ds1": self.ds1(h1),
            "logits_ds2": self.ds2(h2),
        }


def deep_supervision_loss(
    logits: Dict[str, torch.Tensor],
    y: torch.Tensor,
    w: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    logits: dict from model(..., return_all_logits=True)
    y: (B,) int64 class labels
    w: weights for (ds1, ds2, final)
    class_weights: optional tensor shape (n_classes,)
    """
    ce = nn.CrossEntropyLoss(weight=class_weights)
    l1 = ce(logits["logits_ds1"], y)
    l2 = ce(logits["logits_ds2"], y)
    l3 = ce(logits["logits_final"], y)
    return w[0] * l1 + w[1] * l2 + w[2] * l3
