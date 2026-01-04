"""
models_2/adni_smoke_test_multimodal_classifier_headmlp_big_010526.py

- D2 실험용: capacity를 키운 버전(d_model=32, depth=4 사용 권장)
- base MultimodalClassifier(122925_rs)를 감싸고,
  classifier(out_features=num_classes)인 nn.Linear만 찾아 MLP head로 교체한다.
- backbone 내부 out_proj 같은 Linear는 절대 건드리지 않는다.
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models_2.adni_smoke_test_multimodal_classifier_122925_rs import MultimodalClassifier


def _replace_classifier_linear_with_mlp(
    model: nn.Module,
    num_classes: int,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    use_gelu: bool = True,
) -> dict:
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
                target_linear = child  # last match wins

    if target_linear is None:
        return {
            "replaced": False,
            "reason": f"No nn.Linear with out_features == num_classes({num_classes}) found.",
            "target_name": None,
        }

    in_features = target_linear.in_features
    out_features = target_linear.out_features
    hidden = max(16, hidden_mult * in_features)

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


class MultimodalClassifierHeadMLPBig(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        num_classes: int = 3,
        depth: int = 4,
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
            print("🧠 HeadMLPBig patch info:", info)

        if not info.get("replaced", False):
            raise RuntimeError(
                "Failed to replace classifier linear layer. "
                "Need nn.Linear(out_features == num_classes)."
            )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.base(images, tabular)