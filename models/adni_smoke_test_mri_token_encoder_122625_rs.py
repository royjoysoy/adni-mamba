import torch
import torch.nn as nn

class MRITokenEncoder3D(nn.Module):
    """
    Input : (B, 1, 64, 64, 64)
    Output: (B, N_mri, d_model)
    """
    def __init__(self, d_model=16, base_channels=32):
        super().__init__()

        # 64 -> 32 -> 16 -> 8
        self.cnn = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Linear(base_channels * 4, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        feat = self.cnn(x)  # (B, C, 8, 8, 8)
        B, C, D, H, W = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)  # (B, N_mri, C)
        tokens = self.proj(tokens)                # (B, N_mri, d_model)
        tokens = self.out_norm(tokens)
        return tokens
