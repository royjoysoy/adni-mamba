
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
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Paths
# -------------------------
CSV_IN = "/home/ads_sry/royseo/projects/adni-mamba/A_1_19_26/reports/week2/day1_roi_progression/roi_progression_slopes_01_22_26.csv"
OUT_DIR = Path("/home/ads_sry/royseo/projects/adni-mamba/A_1_19_26/reports/week2/day1_roi_progression")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_IN)

roi = "Ventricles"

# -------------------------
# Histogram
# -------------------------
plt.figure()
plt.hist(df[f"slope_{roi}"].dropna(), bins=50)
plt.title(f"{roi} slope distribution")
plt.xlabel("ΔROI / Month")
plt.ylabel("Count")

hist_out = OUT_DIR / f"hist_slope_{roi}_01_22_26.png"
plt.savefig(hist_out, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# Scatter plot
# -------------------------
plt.figure()
plt.scatter(df["Month_bl"], df[f"slope_{roi}"], alpha=0.3)
plt.xlabel("Month_bl")
plt.ylabel(f"{roi} slope")
plt.title(f"{roi} progression rate")

scatter_out = OUT_DIR / f"scatter_slope_{roi}_vs_monthbl_01_22_26.png"
plt.savefig(scatter_out, dpi=300, bbox_inches="tight")
plt.close()

print(f"[Saved] {hist_out}")
print(f"[Saved] {scatter_out}")
