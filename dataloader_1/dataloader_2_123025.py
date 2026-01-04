# 12-20-2025
# dataloader with proper tabular normalization

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
    "Sex",      # F / M
    "PTMARRY",  # categorical
    "APOE4"     # 0 / 1 / 2
]

ALL_TAB_COLS = CONT_COLS + CAT_COLS


# =========================
# Utility: compute z-score stats
# =========================

def compute_zscore_stats(df, cols):
    stats = {}
    for c in cols:
        mu = df[c].mean()
        std = df[c].std()
        if std < 1e-6 or np.isnan(std):
            std = 1.0
        stats[c] = (mu, std)
    return stats


# =========================
# Dataset
# =========================

class ADNI_Multimodal_Dataset(Dataset):
    def __init__(self, dataframe, img_dir, stats):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.stats = stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -----------------
        # 1. Image loading
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
        # 2. Tabular preprocessing
        # -----------------
        tab = []

        # continuous → z-score
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

        tabular = torch.tensor(tab, dtype=torch.float32)

        # -----------------
        # 3. Label
        # -----------------
        label = torch.tensor(int(row["DX2"]), dtype=torch.long)

        return {
            "image": img,
            "tabular": tabular,
            "label": label
        }


# =========================
# DataLoader preparation
# =========================

def prepare_adni_loaders(csv_path, img_dir, batch_size=2):
    df = pd.read_csv(csv_path)

    # 필수 컬럼 결측 제거
    cols_to_check = ["DX2", "Sex", "Age", "Image Data ID", "Subject"]
    df = df.dropna(subset=cols_to_check)

    # -----------------
    # Patient-wise split
    # -----------------
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(splitter.split(df, groups=df["Subject"]))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    val_test_splitter = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(val_test_splitter.split(temp_df, groups=temp_df["Subject"]))

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

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
# Smoke test
# =========================

if __name__ == "__main__":
    CSV = "/home/ads_sry/royseo/projects/adni-mamba/data_0/ADNI-smoke-test-list-wise-deletion-rs_copy.csv"
    IMG = "/home/ads_sry/royseo/data"

    t_loader, v_loader, s_loader = prepare_adni_loaders(CSV, IMG, batch_size=2)

    for b in t_loader:
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
        break