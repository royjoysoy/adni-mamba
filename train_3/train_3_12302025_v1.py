import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 0. 프로젝트 경로 설정
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ============================================================
# 1. 모듈 import
# ============================================================
from dataloader_1.dataloader_2_123025 import prepare_adni_loaders
from models_2.adni_smoke_test_multimodal_classifier_122925_rs import (
    MultimodalClassifier
)

# ============================================================
# NaN hook 함수 (⚠️ train 밖에 정의)
# ============================================================
def add_nan_hooks(model):
    def make_hook(name):
        def hook(module, inp, out):
            if torch.is_tensor(out) and not torch.isfinite(out).all():
                print(f"\n❌ NaN 발생 레이어: {name} ({module.__class__.__name__})")
                print(
                    "   out min/max:",
                    out.min().item(),
                    out.max().item(),
                    "dtype:", out.dtype
                )
                raise RuntimeError("NaN detected in forward")
        return hook

    for name, m in model.named_modules():
        if isinstance(
            m,
            (nn.Linear, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm3d),
        ):
            m.register_forward_hook(make_hook(name))


# ============================================================
# 2. Train function
# ============================================================
def train():

    # ----------------------------
    # 기본 설정
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 2
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 3

    CSV_PATH = os.path.join(
        project_root, "data_0/ADNI-smoke-test-list-wise-deletion-rs_copy.csv"
    )
    IMG_DIR = "/home/ads_sry/royseo/data"

    print(f"🚀 Device: {device}")

    # ----------------------------
    # 데이터 로더
    # ----------------------------
    print("📦 데이터를 불러오는 중...")
    train_loader, val_loader, _ = prepare_adni_loaders(
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE
    )

    # ----------------------------
    # 모델
    # ----------------------------
    model = MultimodalClassifier(
        d_model=16,
        num_classes=NUM_CLASSES,
        depth=2
    ).to(device)

    # ✅ 여기서 hook 붙인다 (이 위치가 핵심)
    add_nan_hooks(model)

    # ----------------------------
    # Loss / Optimizer
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("🚀 학습 시작")
    print("-" * 60)

    # ========================================================
    # Training loop
    # ========================================================
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):

            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["label"].to(device)

            if labels.dtype != torch.long:
                labels = labels.long()

            if epoch == 0 and batch_idx == 0:
                print("🧪 DTYPE CHECK")
                print("   images dtype :", images.dtype)
                print("   tabular dtype:", tabular.dtype)
                print("   labels dtype :", labels.dtype)
                print("-" * 60)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images, tabular)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.0 * correct / total:.1f}%"
                )

        print(
            f"==> Epoch {epoch+1} Summary | "
            f"Loss: {running_loss / len(train_loader):.4f} | "
            f"Acc: {100.0 * correct / total:.1f}%"
        )
        print("-" * 60)


# ============================================================
# 3. Run
# ============================================================
if __name__ == "__main__":
    train()
