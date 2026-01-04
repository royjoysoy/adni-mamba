"""
train_3_testA_overfitting_010426.py

[Purpose]
- Test A: Overfitting sanity-check 전용 학습 스크립트.
- 목표: train set을 매우 작은 N개(예: 64개)로 줄인 뒤,
  모델이 그 작은 데이터에 대해 "과적합(훈련 정확도 80~100%)"이 가능한지 확인한다.

[What this file does]
1) 프로젝트 루트를 sys.path에 추가하여 모듈 import가 되게 한다.
2) Test A 전용 dataloader(Overfit shrink + tabular normalization 적용)를 불러온다.
3) 모델(MultimodalClassifier)을 만들고,
4) 작은 train subset(overfit_n)만으로 여러 epoch 학습한다.
5) (옵션) NaN hook을 걸어 NaN이 생기면 어떤 레이어에서 터지는지 즉시 출력한다.

[How to run]
- 프로젝트 루트(adni-mamba)에서:
    python train_3/train_3_testA_overfitting_010426.py
  또는 이 파일이 있는 위치에서 실행.

[Important]
- 이 테스트에서 validation/test 성능은 보지 않는다.
- 목적은 오직 하나: "학습 자체 가능 여부" 확인.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 0. 프로젝트 최상위 경로를 파이썬 경로에 추가
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


# ============================================================
# 1. 모듈 import
# ============================================================
from dataloader_1.dataloader_2_testA_overfitting_010426 import prepare_adni_loaders
from models_2.adni_smoke_test_multimodal_classifier_122925_rs import MultimodalClassifier


# ============================================================
# 2. Optional: NaN hook
# ============================================================
def add_nan_hooks(model: nn.Module):
    def make_hook(name):
        def hook(module, inp, out):
            if torch.is_tensor(out) and not torch.isfinite(out).all():
                print(f"\n❌ NaN 발생 레이어: {name} ({module.__class__.__name__})")
                print("   out min/max:", out.min().item(), out.max().item(), "dtype:", out.dtype)
                raise RuntimeError("NaN detected in forward")
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm3d)):
            m.register_forward_hook(make_hook(name))


# ============================================================
# 3. Train (Test A)
# ============================================================
def train_testA_overfit():

    # ----------------------------
    # 기본 설정 (Test A 고정값)
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 2
    EPOCHS = 40
    LEARNING_RATE = 1e-4  # Test A는 "학습 가능 여부" 확인이 목적이라 우선 그대로 둠
    NUM_CLASSES = 3

    OVERFIT_N = 64  # ✅ Test A 핵심: train 샘플 수 (50~100 권장, 여기서는 64)

    CSV_PATH = os.path.join(project_root, "data_0/ADNI-smoke-test-list-wise-deletion-rs_copy.csv")
    IMG_DIR = "/home/ads_sry/royseo/data"

    print(f"🚀 Device: {device}")
    print(f"🧪 Test A Overfit Mode: OVERFIT_N={OVERFIT_N}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
    print("-" * 70)

    # ----------------------------
    # 데이터 로더 (train만 사용)
    # ----------------------------
    print("📦 데이터를 불러오는 중 (overfit train subset)...")
    train_loader, _, _ = prepare_adni_loaders(
        CSV_PATH,
        IMG_DIR,
        batch_size=BATCH_SIZE,
        overfit_n=OVERFIT_N,
        seed=42
    )

    # ----------------------------
    # 모델 정의
    # ----------------------------
    model = MultimodalClassifier(d_model=16, num_classes=NUM_CLASSES, depth=2).to(device)

    # NaN hook (원하면 True로 유지)
    ENABLE_NAN_HOOK = True
    if ENABLE_NAN_HOOK:
        add_nan_hooks(model)

    # ----------------------------
    # 손실함수 / 옵티마이저
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------
    # Training loop (train only)
    # ----------------------------
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
                print("-" * 70)

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

            # overfit test는 로그가 너무 많으면 보기 힘드니 10스텝마다만 찍음
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.1f}%"
                )

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_acc = 100.0 * correct / max(1, total)

        print(f"==> Epoch {epoch+1} Summary | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}%")
        print("-" * 70)


if __name__ == "__main__":
    train_testA_overfit()