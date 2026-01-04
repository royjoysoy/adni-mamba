"""
train_3_testC_ablation_image_only_single_batch_overfit_010426.py

[Purpose]
- IMAGE-ONLY + Single-batch overfit test (mixed-label batch 우선 선택)
- 목표: "딱 1 배치(2샘플)"만 100% 외울 수 있는지 확인
- 가장 강력한 sanity-check:
  - mixed-label 배치도 못 외우면: head/스케일/학습 구조 문제 가능성 ↑
  - mixed-label 배치는 외우는데 64개는 못 외우면: capacity 부족 가능성 ↑

[What this script does]
1) overfit_n=64로 train subset을 만든다.
2) train subset의 label 분포를 출력한다.
3) train_loader에서 "서로 다른 클래스가 섞인 배치(mixed-label)"를 우선적으로 찾아 고정한다.
   - 없으면 첫 배치를 fallback으로 사용한다.
4) tabular 입력은 zeros로 만들고, model의 tab_embed 가중치/바이어스를 0으로 만들고 freeze하여
   tabular branch 영향이 완전히 0이 되도록 한다.
5) 고정된 배치 하나만 STEPS번 반복 학습해서 Fixed-batch Acc가 100%로 가는지 확인한다.
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
from models_2.adni_smoke_test_multimodal_classifier_122925_rs import MultimodalClassifier


# ============================================================
# 2. Tabular branch 영향 제거
# ============================================================
def disable_tabular_branch(model: nn.Module):
    """
    tab/tabular 관련 파라미터를 찾아 0으로 만들고 freeze.
    현재 모델에서는 tab_embed.weight/bias를 잡는 것이 목표.
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


@torch.no_grad()
def batch_acc(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    pred = outputs.argmax(dim=1)
    return (pred == labels).float().mean().item() * 100.0


def select_mixed_label_batch(train_loader):
    """
    train_loader를 훑어서 label이 서로 다른 샘플이 포함된 배치(mixed-label)를 우선 선택.
    없으면 첫 배치를 fallback.
    """
    first_batch = None
    for b in train_loader:
        if first_batch is None:
            first_batch = b
        y = b["label"].cpu().numpy().tolist()
        if len(set(y)) > 1:
            return b, True  # mixed-label found
    return first_batch, False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 설정 ======
    BATCH_SIZE = 2
    OVERFIT_N = 64
    LR = 1e-4
    STEPS = 500
    PRINT_EVERY = 20
    NUM_CLASSES = 3

    CSV_PATH = os.path.join(project_root, "data_0/ADNI-smoke-test-list-wise-deletion-rs_copy.csv")
    IMG_DIR = "/home/ads_sry/royseo/data"

    print(f"🚀 Device: {device}")
    print(f"🧪 Single-batch overfit (mixed-label preferred) | STEPS={STEPS}, LR={LR}, OVERFIT_N={OVERFIT_N}")
    print("-" * 70)

    # ====== Data ======
    train_loader, _, _ = prepare_adni_loaders(
        CSV_PATH,
        IMG_DIR,
        batch_size=BATCH_SIZE,
        overfit_n=OVERFIT_N,
        seed=42
    )

    # label 분포 출력
    cnt = Counter()
    for b in train_loader:
        cnt.update(b["label"].cpu().numpy().tolist())
    print("📊 LABEL COUNT (train subset):", dict(cnt))
    print("-" * 70)

    # mixed-label batch 선택
    fixed_batch, is_mixed = select_mixed_label_batch(train_loader)
    fixed_labels = fixed_batch["label"].cpu().numpy().tolist()
    print(f"🧪 FIXED BATCH labels: {fixed_labels} | mixed-label: {is_mixed}")
    print("-" * 70)

    # ====== Tensors ======
    images = fixed_batch["image"].to(device)
    labels = fixed_batch["label"].to(device).long()
    tabular = torch.zeros_like(fixed_batch["tabular"]).to(device)

    # ====== Model ======
    model = MultimodalClassifier(d_model=16, num_classes=NUM_CLASSES, depth=2).to(device)
    disable_tabular_branch(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ====== Train (fixed batch only) ======
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = model(images, tabular)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if step % PRINT_EVERY == 0 or step == 1:
            with torch.no_grad():
                acc = batch_acc(outputs, labels)
            print(f"Step [{step}/{STEPS}] Loss: {loss.item():.4f} | Fixed-batch Acc: {acc:.1f}%")

    print("-" * 70)
    print("✅ Done. If mixed-label fixed-batch Acc doesn't reach 100%, head/scale/training issue is likely.")


if __name__ == "__main__":
    main()