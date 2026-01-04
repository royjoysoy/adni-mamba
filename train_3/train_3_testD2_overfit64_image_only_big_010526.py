"""
train_3/train_3_testD2_overfit64_image_only_big_DEBUG_010526.py

[Purpose]
- D2 Overfit64 IMAGE-ONLY (BIG model) + DEBUG version
- 목표:
  1) overfit_n=64에서도 acc가 40.6%에 고정되는 이유를 "증거로" 확정
  2) single-class collapse인지, 입력/라벨 매칭 문제인지, 입력이 거의 동일한지 바로 판별

[What this script adds vs non-debug version]
- (A) First batch image stats per-sample: mean/std/min/max
- (B) First batch img_path 출력 (dataloader가 img_path를 반환할 때만)
- (C) Epoch-end:
    - pred histogram / label histogram
    - logits mean/std
    - softmax probability mean (클래스별 평균 확률)
- (D) (Optional) class-weighted CE loss to break trivial collapse (default OFF)

[Assumptions]
- dataloader: dataloader_1/dataloader_2_testA_overfitting_010426.py
  - prepare_adni_loaders(..., overfit_n=64, seed=42) 지원
  - batch dict contains: "image", "tabular", "label"
  - (OPTIONAL) If you added img_path in dataset return, it will print it.

- model: models_2/adni_smoke_test_multimodal_classifier_headmlp_big_010526.py
  - MultimodalClassifierHeadMLPBig(d_model=32, depth=4, num_classes=3)
"""

import sys
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 0) Project root path
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ============================================================
# 1) Imports
# ============================================================
from dataloader_1.dataloader_2_testA_overfitting_010426 import prepare_adni_loaders
from models_2.adni_smoke_test_multimodal_classifier_headmlp_big_010526 import (
    MultimodalClassifierHeadMLPBig
)


# ============================================================
# 2) Utilities
# ============================================================
def disable_tabular_branch(model: nn.Module):
    """
    IMAGE-ONLY: tabular 관련 파라미터를 0으로 만들고 freeze.
    + training loop에서 tabular input도 zeros로 넣는다.
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
def compute_epoch_debug(model: nn.Module, loader, device):
    """
    epoch-end 디버그:
    - pred hist / label hist
    - logits mean/std
    - softmax probability mean
    """
    model.eval()
    pred_cnt = Counter()
    label_cnt = Counter()
    all_logits = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()
        tabular = torch.zeros_like(batch["tabular"]).to(device)

        logits = model(images, tabular)  # [B, C]
        preds = logits.argmax(dim=1)

        pred_cnt.update(preds.detach().cpu().numpy().tolist())
        label_cnt.update(labels.detach().cpu().numpy().tolist())
        all_logits.append(logits.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)  # [N, C]
    probs_mean = torch.softmax(all_logits, dim=1).mean(dim=0)  # [C]

    dbg = {
        "pred_hist": dict(pred_cnt),
        "label_hist": dict(label_cnt),
        "logits_mean": float(all_logits.mean().item()),
        "logits_std": float(all_logits.std().item()),
        "probs_mean": probs_mean.numpy(),
    }
    return dbg


def print_first_batch_debug(batch, device):
    """
    첫 batch에서 입력이 '진짜 서로 다른지' 빠르게 확인.
    - images per-sample mean/std/min/max
    - img_path가 있으면 출력
    """
    images = batch["image"].to(device)
    labels = batch["label"].to(device).long()

    print("🧪 FIRST-BATCH DEBUG")
    print("  labels:", labels.detach().cpu().tolist())
    print("  image stats per-sample:")
    for i in range(images.size(0)):
        x = images[i]
        print(
            f"    sample{i}: mean={x.mean().item():.6f} std={x.std().item():.6f} "
            f"min={x.min().item():.6f} max={x.max().item():.6f}"
        )

    # OPTIONAL: dataloader가 img_path를 주는 경우에만
    if "img_path" in batch:
        print("  img_path:", batch["img_path"])
    else:
        print("  img_path: (not provided by dataloader)")

    print("-" * 70)


# ============================================================
# 3) Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # D2 config
    # ----------------------------
    BATCH_SIZE = 2
    OVERFIT_N = 64
    EPOCHS = 30          # 디버그용: 일단 30 (필요하면 120으로 다시)
    LR = 1e-3
    WEIGHT_DECAY = 0.0
    NUM_CLASSES = 3

    # ----------------------------
    # Paths
    # ----------------------------
    CSV_PATH = os.path.join(project_root, "data_0/ADNI-smoke-test-list-wise-deletion-rs_copy.csv")
    IMG_DIR = "/home/ads_sry/royseo/data"

    print(f"🚀 Device: {device}")
    print(
        f"🧪 D2 Overfit64 IMAGE-ONLY BIG DEBUG | d_model=32 depth=4 | "
        f"epochs={EPOCHS} bs={BATCH_SIZE} lr={LR} wd={WEIGHT_DECAY}"
    )
    print("-" * 70)

    # ----------------------------
    # Loaders (train subset only)
    # ----------------------------
    train_loader, _, _ = prepare_adni_loaders(
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, overfit_n=OVERFIT_N, seed=42
    )

    # Label distribution check
    cnt = Counter()
    for b in train_loader:
        cnt.update(b["label"].cpu().numpy().tolist())
    print("📊 LABEL COUNT (train subset):", dict(cnt))
    print("-" * 70)

    # ----------------------------
    # Model (BIG + HeadMLP)
    # ----------------------------
    model = MultimodalClassifierHeadMLPBig(
        d_model=32,
        num_classes=NUM_CLASSES,
        depth=4,
        head_hidden_mult=4,
        head_dropout=0.0,
        head_use_gelu=True,
        verbose=True,
    ).to(device)

    # IMAGE-ONLY
    disable_tabular_branch(model)

    # ----------------------------
    # Loss / Optimizer
    # ----------------------------
    USE_CLASS_WEIGHT = False  # ✅ collapse 깨기용 (일단 False로)
    if USE_CLASS_WEIGHT:
        # label count: {0:22,1:26,2:16} 예시
        # 일반화된 방식: 현재 subset label count로 weight 계산
        label_hist = dict(cnt)
        w = torch.tensor(
            [1.0 / max(1, label_hist.get(c, 1)) for c in range(NUM_CLASSES)],
            device=device,
            dtype=torch.float32,
        )
        w = w / w.mean()
        print("🧪 Using class-weighted CE. weight =", w.detach().cpu().tolist())
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # ----------------------------
    # Train loop
    # ----------------------------
    first_batch_debug_printed = False

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            tabular = torch.zeros_like(batch["tabular"]).to(device)  # IMAGE-ONLY

            # First-batch debug (only once)
            if not first_batch_debug_printed:
                print_first_batch_debug(batch, device)
                first_batch_debug_printed = True

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

        # Epoch-end debug
        dbg = compute_epoch_debug(model, train_loader, device)

        print(f"==> Epoch {epoch+1} Summary | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}%")
        print("    Pred hist :", dbg["pred_hist"])
        print("    Label hist:", dbg["label_hist"])
        print(f"    LOGITS mean/std: {dbg['logits_mean']:.6f} / {dbg['logits_std']:.6f}")
        print("    PROB mean:", [float(f"{x:.4f}") for x in dbg["probs_mean"]])
        print("-" * 70)


if __name__ == "__main__":
    main()