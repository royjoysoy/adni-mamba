"""
ChatGPT + Roy Seo 1-22-2026
[한국어 설명]
이 스크립트는 ADNI tabular CSV에서 ROI(뇌 영역 부피) 기반의 "진행 속도(progression speed)" 지표를 만듭니다.
각 ROI를 ICV로 정규화한 뒤(ROI/ICV), baseline ROI와의 차이(ΔROI)를 계산하고,
Month_bl(베이스라인 이후 경과 개월 수)로 나누어 slope(월당 변화율)를 산출합니다.

- 목적:
  1) ROI를 모델 입력으로 쓰지 않고,
  2) 모델 결과(CNN/Mamba)가 '진행 속도'를 잡는지 검증할 수 있는 ground-truth proxy(대리 지표)를 만들기 위함입니다.

- 결측치 처리:
  ROI follow-up 값은 imputation하지 않습니다. (결측이 있는 row는 해당 ROI의 slope가 NaN으로 남습니다)
  ICV는 missing=0이라고 가정합니다.

[English Description]
This script builds ROI-based progression-speed proxies from an ADNI merged tabular CSV.
For each ROI, it performs ICV normalization (ROI/ICV), computes the baseline-referenced change (ΔROI),
and derives a per-month slope by dividing ΔROI by Month_bl (months since baseline).

- Goal:
  Create ROI-based progression proxies (not used as model inputs) for validating whether CNN/Mamba captures
  subject-specific disease progression speed.

- Missingness:
  No imputation is performed for follow-up ROI values. Missing ROI values lead to NaN slopes for that ROI.
  ICV is assumed to have 0 missing values.

Inputs:
  - CSV: /home/ads_sry/royseo/projects/adni-mamba/data_0/ADNI_master_merged_12-17-2025.csv

Outputs:
  - CSV: /home/ads_sry/royseo/projects/adni-mamba/A_1_19_26/reports/week2/day1_roi_progression/roi_progression_slopes_01_22_26.csv
"""

import pandas as pd
import numpy as np

CSV = "/home/ads_sry/royseo/projects/adni-mamba/data_0/ADNI_master_merged_12-17-2025.csv"
OUT = "/home/ads_sry/royseo/projects/adni-mamba/A_1_19_26/reports/roi_progression_slopes.csv"

ROIS = [
    "Ventricles",
    "WholeBrain",
    "Hippocampus",
    "Entorhinal",
]

df = pd.read_csv(CSV, low_memory=False)

# 필수 컬럼
keep = ["RID", "Month_bl", "ICV", "ICV_bl"]
for r in ROIS:
    keep += [r, f"{r}_bl"]

df = df[keep].copy()

# ICV 0-missing 가정
for r in ROIS:
    df[f"{r}_norm"] = df[r] / df["ICV"]
    df[f"{r}_bl_norm"] = df[f"{r}_bl"] / df["ICV"]

    df[f"delta_{r}"] = df[f"{r}_norm"] - df[f"{r}_bl_norm"]
    df[f"slope_{r}"] = df[f"delta_{r}"] / df["Month_bl"]

# Month_bl == 0 제거 (baseline 자체)
df = df[df["Month_bl"] > 0]

df.to_csv(OUT, index=False)
print(f"[Wrote] {OUT}")