"""
File: models_rs/cae_3d_01_24_2026.py
Date: 01_24_2026
Purpose:
  - 3D Convolutional AutoEncoder (CAE) for unsupervised pretraining.
  - Input tensor shape: (B, 1, 64, 64, 64) skull-stripped T1
  - Output: reconstruction x_hat same shape as input.

Optional:
  - masked reconstruction loss to focus only on brain voxels.
    * If you have a brain mask tensor: mask shape (B,1,64,64,64) with {0,1}

How to use (typical):
  from models_rs.cae_3d_01_24_2026 import CAE3D, masked_mse

Training objective:
  - minimize MSE or masked MSE between x_hat and x
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE3D(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()

        # Encoder (match DSA3DCNN encoder blocks)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 96->48
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 48->24
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 24->12
        )

        # Decoder (upsample + conv)
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 12->24
            nn.Conv3d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 24->48
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),  # 48->96
            nn.Conv3d(32, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc3(self.enc2(self.enc1(x)))
        x_hat = self.dec1(self.dec2(self.dec3(z)))
        return x_hat


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    x_hat, x: (B,1,96,96,96)
    mask: optional (B,1,96,96,96) with {0,1} (float/bool)
    """
    if mask is None:
        return F.mse_loss(x_hat, x)
    diff = (x_hat - x) * mask
    return (diff.pow(2).sum() / (mask.sum() + 1e-8))