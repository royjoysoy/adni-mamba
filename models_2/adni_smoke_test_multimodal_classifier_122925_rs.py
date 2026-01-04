import sys
import os

# 1. 현재 파일의 절대 경로를 기준으로 프로젝트 루트를 찾아서 '최우선' 등록
current_file_path = os.path.dirname(os.path.abspath(__file__))  # .../models_2
project_root = os.path.dirname(current_file_path)              # .../adni-mamba

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

# TokenAssembler import (Absolute -> Fallback)
try:
    from models_2.adni_smoke_test_multimodal_token_assembler_122625_rs import MultimodalTokenAssembler
    print("✅ TokenAssembler 임포트 성공 (Absolute Import)")
except ImportError:
    sys.path.insert(0, current_file_path)
    from adni_smoke_test_multimodal_token_assembler_122625_rs import MultimodalTokenAssembler
    print("✅ TokenAssembler 임포트 성공 (Fallback Import)")

# ✅ mamba_ssm의 Mamba layer import (버전 차이 대비)
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    from mamba_ssm.modules.mamba2 import Mamba  # type: ignore


# --- 모델 구조 (동일) ---

class MRIEncoder(nn.Module):
    """64x64x64 이미지를 받아서 (512, d_model) 크기의 토큰 시퀀스로 변환"""
    def __init__(self, d_model=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv3d(16, d_model, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x) 
        x = x.flatten(2) 
        return x.transpose(1, 2) 

class RepoMambaBackbone(nn.Module):
    """
    mamba_ssm의 Mamba 레이어를 depth 만큼 쌓아서 encoder처럼 사용.
    입력/출력: (B, L, D)
    """
    def __init__(self, d_model=16, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

class MultimodalClassifier(nn.Module):
    def __init__(self, d_model=16, num_classes=3, depth=2):
        super().__init__()
        self.mri_encoder = MRIEncoder(d_model=d_model)
        self.tab_embed = nn.Linear(10, 10 * d_model) 
        self.assembler = MultimodalTokenAssembler()
        self.backbone = RepoMambaBackbone(d_model=d_model, depth=depth)
        self.head = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, img, tab):
        tokens_mri = self.mri_encoder(img)
        tokens_tab = self.tab_embed(tab).view(-1, 10, self.d_model)
        tokens = self.assembler(tokens_mri, tokens_tab)
        tokens = self.backbone(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalClassifier(d_model=16, num_classes=3, depth=2).to(device)
    
    dummy_img = torch.randn(2, 1, 64, 64, 64).to(device)
    dummy_tab = torch.randn(2, 10).to(device)
    
    output = model(dummy_img, dummy_tab)
    print(f"✅ Output shape: {output.shape}") 
    print("✅ Model 보정 완료 및 테스트 성공!")