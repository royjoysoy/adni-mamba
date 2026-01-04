"""
models_2/adni_smoke_test_multimodal_classifier_headmlp_010526.py

- 기존 MultimodalClassifier(122925_rs)를 그대로 사용하면서,
  "classifier 역할(출력 차원 = num_classes)"을 하는 nn.Linear만 찾아 MLP head로 교체하는 실험용 모델.

왜 이렇게 하냐?
- Mamba block 내부(out_proj 등)는 Linear여야 하고 weight를 직접 참조한다.
- 따라서 backbone 내부 Linear를 건드리면 깨진다.
- 안전하게 "out_features == num_classes" 인 Linear만 교체한다.

NOTE
- train 스크립트에서 import해서 쓰는 용도.
- 단독 실행도 가능하도록 project root를 sys.path에 추가.
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

# ============================================================
# 0) project root를 sys.path에 추가 (단독 실행 대비)
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅ 원본 모델 import
from models_2.adni_smoke_test_multimodal_classifier_122925_rs import MultimodalClassifier


def _replace_classifier_linear_with_mlp(
    model: nn.Module,
    num_classes: int,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    use_gelu: bool = True,
) -> dict:
    """
    model 안에서 out_features == num_classes 인 nn.Linear를 찾아 (가장 마지막 발견된 것)
    MLP head로 교체한다.

    Returns:
      info dict with replacement details.
    """
    target_parent = None
    target_attr = None
    target_name = None
    target_linear = None

    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child.out_features == num_classes:
                target_parent = module
                target_attr = child_name
                target_name = f"{module_name}.{child_name}" if module_name else child_name
                target_linear = child  # keep updating -> last match wins

    if target_linear is None:
        return {
            "replaced": False,
            "reason": f"No nn.Linear with out_features == num_classes({num_classes}) found.",
            "target_name": None,
        }

    in_features = target_linear.in_features
    out_features = target_linear.out_features  # == num_classes
    hidden = max(8, hidden_mult * in_features)

    act = nn.GELU() if use_gelu else nn.ReLU(inplace=False)

    mlp = nn.Sequential(
        nn.Linear(in_features, hidden),
        act,
        nn.Dropout(p=dropout),
        nn.Linear(hidden, out_features),
    )

    setattr(target_parent, target_attr, mlp)

    return {
        "replaced": True,
        "target_name": target_name,
        "in_features": in_features,
        "out_features": out_features,
        "hidden": hidden,
        "dropout": dropout,
        "activation": "GELU" if use_gelu else "ReLU",
    }


class MultimodalClassifierHeadMLP(nn.Module):
    """
    원본 MultimodalClassifier를 감싸서,
    classifier(out_features=num_classes)만 MLP head로 바꾼 버전.
    """

    def __init__(
        self,
        d_model: int = 16,
        num_classes: int = 3,
        depth: int = 2,
        head_hidden_mult: int = 4,
        head_dropout: float = 0.0,
        head_use_gelu: bool = True,
        verbose: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.base = MultimodalClassifier(
            d_model=d_model,
            num_classes=num_classes,
            depth=depth,
        )

        info = _replace_classifier_linear_with_mlp(
            self.base,
            num_classes=num_classes,
            hidden_mult=head_hidden_mult,
            dropout=head_dropout,
            use_gelu=head_use_gelu,
        )

        if verbose:
            print("🧠 HeadMLP patch info:", info)

        if not info.get("replaced", False):
            raise RuntimeError(
                "Failed to replace classifier linear layer. "
                "Patch criteria: nn.Linear(out_features == num_classes). "
                "Please inspect base model head naming."
            )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.base(images, tabular)


# (선택) 단독 실행 smoke test
if __name__ == "__main__":
    m = MultimodalClassifierHeadMLP(verbose=True)
    print("✅ model instantiated OK")