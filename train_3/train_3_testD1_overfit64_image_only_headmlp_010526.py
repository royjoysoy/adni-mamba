"""
train_3/train_3_testD1_overfit64_image_only_headmlp_010526.py

[Goal]
- D1 실험: overfit_n=64, IMAGE-ONLY 조건에서
  "head를 MLP로 강화"하면 학습(암기)이 되는지 확인한다.

[Setup]
- dataloader: 기존 dataloader_2_testA_overfitting_010426.py 그대로 사용
- overfit_n=64 subset으로 train만 사용
- tabular 영향 완전 제거:
  1) 입력 tabular = zeros
  2) 모델 내부 tab_embed.weight/bias를 0으로 만들고 freeze

[Expected]
- 이 설정에서 train acc가 80~100%로 올라가면:
  -> 기존 head가 너무 약했던 것이 주요 원인일 가능성 큼
"""

import sys
import os
from collections import Counter

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
from dataloader_1.dataloader_2_testA_overfitting_010426 import prepare_adni_loaders
from models_2.adni_smoke_test_multimodal_classifier_headmlp_010526 import MultimodalClassifierHeadMLP


def disable_tabular_branch(model: nn.Module):
    """
    tab_* 파라미터를 찾아 0으로 만들고 freeze하여 tabular 영향 제거.
    지금까지 관찰된 이름은 tab_embed.(weight|bias)라서 이걸 잡아낸다.
    """
    key_substrings = ("tab", "tabular", "meta", "clinical", "demographic", "covariate")
    disabled = []

    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in key_substrings):
            with torch.no_grad():
                p.zero_()
            p.requires_grad_(False)
            disabled.append(name)

    print("\n🧪 IMAGE-ONLY: disabled tabular-related params count =", len(disabled))
    for n in disabled:
        print("   -", n)
    print("-" * 70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 실험 설정
    # ----------------------------
    BATCH_SIZE = 2
    OVERFIT_N = 64
    EPOCHS = 80
    LR = 1e-4
    NUM_CLASSES = 3

    CSV_PATH = os.path.join(project_root, "data_0/ADNI-smoke-test-list-wise-deletion-rs_copy.csv")
    IMG_DIR = "/home/ads_sry/royseo/data"

    print(f"🚀 Device: {device}")
    print(f"🧪 D1 Overfit64 IMAGE-ONLY + HeadMLP | overfit_n={OVERFIT_N}, epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LR}")
    print("-" * 70)

    # ----------------------------
    # 데이터 로더 (train subset만 사용)
    # ----------------------------
    train_loader, _, _ = prepare_adni_loaders(
        CSV_PATH,
        IMG_DIR,
        batch_size=BATCH_SIZE,
        overfit_n=OVERFIT_N,
        seed=42
    )

    # label 분포 확인
    cnt = Counter()
    for b in train_loader:
        cnt.update(b["label"].cpu().numpy().tolist())
    print("📊 LABEL COUNT (train subset):", dict(cnt))
    print("-" * 70)

    # ----------------------------
    # 모델 (HeadMLP 버전)
    # ----------------------------
    model = MultimodalClassifierHeadMLP(
        d_model=16,
        num_classes=NUM_CLASSES,
        depth=2,
        head_hidden_mult=4,   # head capacity ↑
        head_dropout=0.0,     # overfit 테스트니까 dropout 0
        head_use_gelu=True,
        verbose=True
    ).to(device)

    # tabular 영향 제거
    disable_tabular_branch(model)

    # optimizer는 학습 가능한 파라미터만
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ----------------------------
    # Train loop (train만)
    # ----------------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()

            # ✅ tabular은 0으로 입력
            tabular = torch.zeros_like(batch["tabular"]).to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

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
    main()