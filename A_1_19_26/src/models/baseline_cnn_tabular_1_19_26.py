import torch
import torch.nn as nn

class Baseline3DCNNTabular_1_19_26(nn.Module):
    """
    Day1 baseline model:
      image (B,1,64,64,64) -> small 3D CNN -> global pool -> img embedding
      tabular (B,4) -> concat -> MLP -> logits (B,3)
    """
    def __init__(self, tabular_dim: int, num_classes: int = 3, emb_dim: int = 128):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # 64->32
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), # 32->16
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # 16->8
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),# 8->4
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.img_proj = nn.Linear(128, emb_dim)

        self.head = nn.Sequential(
            nn.Linear(emb_dim + tabular_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = self.enc(image)                 # (B,128,4,4,4)
        x = self.pool(x).flatten(1)         # (B,128)
        x = self.img_proj(x)                # (B,emb_dim)
        x = torch.cat([x, tabular], dim=1)  # (B,emb_dim+tab_dim)
        logits = self.head(x)               # (B,num_classes)
        return logits
