import torch
import torch.nn as nn

from models_rs.adni_smoke_test_multimodal_token_assembler_122625_rs import MultimodalTokenAssembler
from cross_atten.mamba import Mamba, MambaConfig


class RepoMambaBackbone(nn.Module):
    """repo 내부 cross_atten/mamba.py의 Mamba를 token sequence backbone으로 사용 (CPU fallback)"""
    def __init__(self, d_model=16, depth=2):
        super().__init__()
        config = MambaConfig(
            d_model=d_model,
            n_layers=depth,
            use_cuda=True  
        )
        self.mamba = Mamba(config)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.final_norm(self.mamba(x))


class MultimodalClassifier(nn.Module):
    def __init__(self, d_model=16, num_classes=3, depth=2):
        super().__init__()
        self.assembler = MultimodalTokenAssembler()
        self.backbone = RepoMambaBackbone(d_model=d_model, depth=depth)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens_mri, tokens_tab):
        tokens = self.assembler(tokens_mri, tokens_tab)
        tokens = self.backbone(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


if __name__ == "__main__":
    # 1. 장치 설정 (GPU 사용 가능 여부 확인)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 모델 생성 및 GPU로 이동
    model = MultimodalClassifier(d_model=16, num_classes=3, depth=2).to(device)
    model.eval() # 평가 모드

    # 3. 가짜 데이터 생성 (배치 크기 2, MRI 토큰 512개, 수치 토큰 10개)
    # 각 토큰의 차원은 d_model=16
    dummy_mri = torch.randn(2, 512, 16).to(device)
    dummy_tab = torch.randn(2, 10, 16).to(device)

    # 4. 모델 추론
    with torch.no_grad():
        output = model(dummy_mri, dummy_tab)

    # 5. 결과 출력
    print("--- Test Result ---")
    print(f"Input MRI shape: {dummy_mri.shape}")
    print(f"Input Tabular shape: {dummy_tab.shape}")
    print(f"Output Prediction shape: {output.shape} (Batch, Classes)")
    print("Mamba Forward Pass: Successful!")