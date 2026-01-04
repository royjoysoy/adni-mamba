"""
train_3_testC_ablation_image_only_overfitting_010426.py

[Purpose]
- Overfitting sanity-check: IMAGE-ONLY 버전
- 목표: 모델이 "이미지 정보만으로" 작은 train subset(overfit_n=64)을 외울 수 있는지 확인

[Approach]
- MultimodalClassifier는 (images, tabular)를 받는 구조이므로,
  tabular branch의 영향을 완전히 제거하기 위해 2단계 처리:
  1) training loop에서 tabular input을 0으로 세팅
  2) model 내 tabular 관련 파라미터(가중치/바이어스)를 0으로 만들고 freeze
     -> tabular 입력이 0이어도 bias로 영향 생기는 경우까지 차단

[Extra diagnostics added]
- LABEL COUNT 출력:
  - overfit_n=64로 뽑힌 train subset의 class 분포가 심하게 불균형인지 확인
- GRAD NORM SUM 출력:
  - 첫 배치(backward 직후)에서 gradient가 실제로 흐르는지(0에 가깝게 죽어있지 않은지) 확인

[Success criterion]
- train accuracy가 80~100%까지 올라가면:
  -> 모델/백본은 학습 가능. 이후 fusion이 문제인지 따로 분리 가능
- 50% 근처에서 계속 정체면:
  -> 이미지 쪽 표현력/학습 구조(백본/헤드/옵티마이저) 문제가 강하게 의심됨
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
# 3. Tabular branch 제거(영향 0) 유틸
# ============================================================
def disable_tabular_branch(model: nn.Module):
    """
    모델 내부에서 'tab'/'tabular'/'meta'/'clinical' 등 이름을 가진 파라미터를 찾아:
    - weight/bias를 0으로 만들고
    - gradient 업데이트를 막아서(tab branch 영향 0)
    image-only 학습이 되도록 만든다.
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
    if len(disabled) > 0:
        for n in disabled[:30]:
            print("   -", n)
        if len(disabled) > 30:
            print(f"   ... and {len(disabled) - 30} more")
    else:
        print("⚠️ WARNING: tabular-related params not found by name matching.")
        print("   In this case, tabular input will still be zeroed, but bias terms may remain.")
    print("-" * 70)


# ============================================================
# 4. Helper: label count (class distribution) on train subset
# ============================================================
def print_label_count(train_loader):
    cnt = Counter()
    for b in train_loader:
        cnt.update(b["label"].cpu().numpy().tolist())
    print("📊 LABEL COUNT (train subset):", dict(cnt))
    print("-" * 70)


# ============================================================
# 5. Helper: grad norm sum (first batch only)
# ============================================================
def grad_norm_sum(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and (p.grad is not None):
            # L2 norm of this param's grad
            total += p.grad.data.norm(2).item()
    return total


# ============================================================
# 6. Train: IMAGE-ONLY overfitting
# ============================================================
def train_image_only_overfit():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 2
    EPOCHS = 40
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 3
    OVERFIT_N = 64

    CSV_PATH = os.path.join(project_root, "data_0/ADNI-smoke-test-list-wise-deletion-rs_copy.csv")
    IMG_DIR = "/home/ads_sry/royseo/data"

    print(f"🚀 Device: {device}")
    print(f"🧪 IMAGE-ONLY Overfit | OVERFIT_N={OVERFIT_N}, EPOCHS={EPOCHS}, BS={BATCH_SIZE}")
    print("-" * 70)

    print("📦 Loading data (train subset only)...")
    train_loader, _, _ = prepare_adni_loaders(
        CSV_PATH,
        IMG_DIR,
        batch_size=BATCH_SIZE,
        overfit_n=OVERFIT_N,
        seed=42
    )

    # ✅ label 분포 출력 (진단 1)
    print_label_count(train_loader)

    model = MultimodalClassifier(d_model=16, num_classes=NUM_CLASSES, depth=2).to(device)

    # NaN hook (원하면 유지)
    ENABLE_NAN_HOOK = True
    if ENABLE_NAN_HOOK:
        add_nan_hooks(model)

    # ✅ 핵심: tabular branch 영향 제거
    disable_tabular_branch(model)

    criterion = nn.CrossEntropyLoss()

    # requires_grad=True 파라미터만 optimizer에 넣기
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)

    printed_grad_norm = False  # grad norm은 첫 배치에서 1번만 출력

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if labels.dtype != torch.long:
                labels = labels.long()

            # ✅ tabular 입력은 0으로 (shape 맞춰서)
            tabular = batch["tabular"].to(device)
            tabular = torch.zeros_like(tabular)

            if epoch == 0 and batch_idx == 0:
                print("🧪 DTYPE/SHAPE CHECK")
                print("   images:", images.dtype, images.shape)
                print("   tabular:", tabular.dtype, tabular.shape)
                print("   labels :", labels.dtype, labels.shape)
                print("-" * 70)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images, tabular)
            loss = criterion(outputs, labels)

            loss.backward()

            # ✅ grad norm 출력 (진단 2): 첫 배치에서 backward 직후 딱 1번
            if (not printed_grad_norm) and (epoch == 0 and batch_idx == 0):
                gsum = grad_norm_sum(model)
                print(f"🧪 GRAD NORM SUM (first batch, after backward): {gsum:.6f}")
                print("-" * 70)
                printed_grad_norm = True

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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
    train_image_only_overfit()