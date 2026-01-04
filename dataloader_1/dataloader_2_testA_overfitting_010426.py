"""
dataloader_2_testA_overfitting_010426.py

[Purpose]
- Test A: Overfitting sanity-check를 위한 전용 dataloader 스크립트.
- 목표: 모델이 "학습 자체가 가능한지(소량 데이터에 과적합 가능한지)" 확인하기 위해
  train set을 아주 작은 N개 샘플로 줄여서 DataLoader를 만들어준다.

[What this file does]
1) CSV를 읽고 필수 컬럼 결측을 제거한다.
2) Subject(환자) 단위로 patient-wise split(Train/Val/Test = 8:1:1)을 만든다.
3) Tabular 연속 변수는 train set에서만 mean/std를 구해 z-score 정규화한다.
   - val/test에도 train stats를 적용 (leakage 방지)
4) Overfitting test를 위해 train_df를 overfit_n만큼 랜덤 샘플링하여 줄일 수 있다.
5) Dataset에서:
   - Image는 nii.gz를 찾아 로딩하고 intensity z-score 정규화한다.
   - Tabular는:
     * CONT_COLS: z-score 적용
     * CAT_COLS : 인코딩만, 정규화는 하지 않음

[How to use]
- 다른 train 스크립트에서:
    from dataloader_1.dataloader_2_testA_overfitting_010426 import prepare_adni_loaders
  처럼 import해서 사용.

- Overfitting mode:
    train_loader, _, _ = prepare_adni_loaders(..., overfit_n=64)

[Notes]
- 이 파일은 "데이터 로딩/정규화/샘플링"만 담당한다.
- 모델/학습 루프(optimizer, loss 등)는 train 파일에서 한다.
"""

import os
import glob
import pandas as pd
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


# =========================
# Tabular column definition
# =========================

CONT_COLS = [
    "Age", "PTEDUCAT",
    "VSWEIGHT", "VSBPSYS", "VSBPDIA",
    "VSPULSE", "VSRESP"
]

CAT_COLS = [
    "Sex",      # F/M -> 1/0
    "PTMARRY",  # string -> mapped int
    "APOE4"     # 0/1/2 (그대로 사용)
]


# =========================
# Utility: compute z-score stats (train only)
# =========================

def compute_zscore_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    stats = {}
    for c in cols:
        mu = df[c].mean()
        std = df[c].std()
        if std < 1e-6 or np.isnan(std):
            std = 1.0
        stats[c] = (float(mu), float(std))
    return stats


# =========================
# Dataset
# =========================

class ADNI_Multimodal_Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, stats: dict):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.stats = stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # -----------------
        # 1) Image loading
        # -----------------
        img_id = str(int(row["Image Data ID"]))
        search_pattern = os.path.join(self.img_dir, f"*I{img_id}.nii.gz")
        found_files = glob.glob(search_pattern)

        if not found_files:
            raise FileNotFoundError(f"Image ID {img_id} not found in {self.img_dir}")

        img_data = nib.load(found_files[0]).get_fdata()
        img = torch.from_numpy(img_data).float().unsqueeze(0)  # [1, 64, 64, 64]

        # Intensity normalization (z-score)
        img = (img - img.mean()) / (img.std() + 1e-8)

        # -----------------
        # 2) Tabular preprocessing + normalization
        # -----------------
        tab = []

        # continuous -> z-score
        for c in CONT_COLS:
            mu, std = self.stats[c]
            x = (row[c] - mu) / std
            tab.append(x)

        # categorical / ordinal (no normalization)
        # Sex
        tab.append(1.0 if row["Sex"] == "F" else 0.0)

        # PTMARRY
        marry_map = {
            "Married": 0,
            "Divorced": 1,
            "Widowed": 2,
            "Never married": 3,
            "Unknown": 4
        }
        tab.append(marry_map.get(row["PTMARRY"], 4))

        # APOE4
        tab.append(row["APOE4"])

        tabular = torch.tensor(tab, dtype=torch.float32)  # shape [10]

        # -----------------
        # 3) Label
        # -----------------
        label = torch.tensor(int(row["DX2"]), dtype=torch.long)  # 0..2

        return {"image": img, "tabular": tabular, "label": label}


# =========================
# DataLoader preparation
# =========================

def prepare_adni_loaders(
    csv_path: str,
    img_dir: str,
    batch_size: int = 2,
    overfit_n: int | None = None,
    seed: int = 42,
):
    """
    Args:
        csv_path: CSV full path
        img_dir: directory containing nii.gz files
        batch_size: batch size (default=2)
        overfit_n: if not None, train_df를 n개로 줄여 overfitting test 수행
        seed: random seed for splits/sampling

    Returns:
        train_loader, val_loader, test_loader
    """

    df = pd.read_csv(csv_path)

    # 필수 컬럼 결측 제거
    cols_to_check = ["DX2", "Sex", "Age", "Image Data ID", "Subject"]
    df = df.dropna(subset=cols_to_check)

    # -----------------
    # Patient-wise split
    # -----------------
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_idx, temp_idx = next(splitter.split(df, groups=df["Subject"]))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    val_test_splitter = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_idx, test_idx = next(val_test_splitter.split(temp_df, groups=temp_df["Subject"]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # -----------------
    # Overfitting mode: shrink train_df
    # -----------------
    if overfit_n is not None:
        n = min(int(overfit_n), len(train_df))
        train_df = train_df.sample(n=n, random_state=seed).reset_index(drop=True)

    # -----------------
    # Compute tabular stats (train only)
    # -----------------
    train_stats = compute_zscore_stats(train_df, CONT_COLS)

    # -----------------
    # Dataset / Loader
    # -----------------
    train_ds = ADNI_Multimodal_Dataset(train_df, img_dir, stats=train_stats)
    val_ds   = ADNI_Multimodal_Dataset(val_df, img_dir, stats=train_stats)
    test_ds  = ADNI_Multimodal_Dataset(test_df, img_dir, stats=train_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =========================
# Smoke test (optional)
# =========================

if __name__ == "__main__":
    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CSV = os.path.join(BASE, "data_0", "ADNI-smoke-test-list-wise-deletion-rs_copy.csv")
    IMG = "/home/ads_sry/royseo/data"

    # overfit_n=None이면 전체 train set
    t_loader, v_loader, s_loader = prepare_adni_loaders(CSV, IMG, batch_size=2, overfit_n=64)

    b = next(iter(t_loader))
    print("✅ image shape:", b["image"].shape)
    print("✅ tabular shape:", b["tabular"].shape)
    print(
        "TAB stats:",
        b["tabular"].min().item(),
        b["tabular"].max().item(),
        b["tabular"].mean().item(),
        b["tabular"].std().item()
    )
    print("✅ labels:", b["label"])
