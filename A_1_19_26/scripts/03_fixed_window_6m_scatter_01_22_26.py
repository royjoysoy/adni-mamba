"""
ChatGPT + Roy Seo 01-22-2026

[한국어 설명]
이 스크립트는 ADNI 마스터 CSV에서 "6개월 fixed-window" 기반 Ventricles 진행속도 proxy를 계산하고,
(전체 / CN / MCI / AD) 4개의 scatter plot을 저장합니다.

- 방법:
  1) 각 RID별 baseline(Month_bl=0 또는 최소 Month_bl) row 선택
  2) 6개월 방문은 Month_bl이 6에 가장 가까운 row를 선택 (기본 허용범위: 4~8개월)
  3) Ventricles를 ICV로 정규화한 후, baseline 대비 변화량(Δ6m)과 월당 slope_6m=Δ6m/6 계산
  4) scatter: x=선택된 실제 Month_bl(대략 6), y=slope_6m

- 목적:
  Month_bl이 매우 작은 방문(1~2개월)에서 slope가 튀는 문제를 줄이기 위한 robustness check.

Inputs:
  - /home/ads_sry/royseo/projects/adni-mamba/data_0/ADNI_master_merged_12-17-2025.csv

Outputs:
  - CSV:
    /home/ads_sry/royseo/projects/adni-mamba/A_1_19_26/reports/week2/day1_roi_progression/fixed6m_ventricles_slopes_01_22_26.csv
  - PNGs:
    fixed6m_scatter_ALL_01_22_26.png
    fixed6m_scatter_CN_01_22_26.png
    fixed6m_scatter_MCI_01_22_26.png
    fixed6m_scatter_AD_01_22_26.png
"""

import matplotlib
matplotlib.use("Agg")  # headless server safe

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_CSV = "/home/ads_sry/royseo/projects/adni-mamba/data_0/ADNI_master_merged_12-17-2025.csv"

OUT_DIR = Path("/home/ads_sry/royseo/projects/adni-mamba/A_1_19_26/reports/week2/day1_roi_progression")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "fixed6m_ventricles_slopes_01_22_26.csv"

# 6개월 타겟과 허용 범위(robustness check)
TARGET_M = 6.0
TOL_M = 2.0  # 6±2개월 -> [4,8]
MIN_M = TARGET_M - TOL_M
MAX_M = TARGET_M + TOL_M

# 색깔(헷갈리지 않게 각 plot마다 다르게)
COLORS = {
    "ALL": "black",
    "CN": "tab:blue",
    "MCI": "tab:orange",
    "AD": "tab:red",
}

LABELS = {0: "CN", 1: "MCI", 2: "AD"}

def pick_baseline(g: pd.DataFrame) -> pd.Series:
    # baseline은 Month_bl==0을 우선, 없으면 가장 작은 Month_bl
    g2 = g.copy()
    g2["Month_bl"] = pd.to_numeric(g2["Month_bl"], errors="coerce")
    g2 = g2.dropna(subset=["Month_bl"])
    if (g2["Month_bl"] == 0).any():
        return g2.loc[g2["Month_bl"].eq(0)].sort_index().iloc[0]
    return g2.sort_values("Month_bl").iloc[0]

def pick_near_6m(g: pd.DataFrame) -> pd.Series | None:
    g2 = g.copy()
    g2["Month_bl"] = pd.to_numeric(g2["Month_bl"], errors="coerce")
    g2 = g2.dropna(subset=["Month_bl"])
    # baseline(0개월)은 제외하고 4~8개월만 후보
    cand = g2[(g2["Month_bl"] > 0) & (g2["Month_bl"] >= MIN_M) & (g2["Month_bl"] <= MAX_M)]
    if len(cand) == 0:
        return None
    cand = cand.assign(_dist=(cand["Month_bl"] - TARGET_M).abs())
    return cand.sort_values(["_dist", "Month_bl"]).iloc[0]

def main():
    df = pd.read_csv(IN_CSV, low_memory=False)

    # 필요한 컬럼 체크
    need = ["RID", "DX2", "Month_bl", "Ventricles", "Ventricles_bl", "ICV"]
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns: {missing_cols}")

    # 숫자형 처리
    df["DX2"] = pd.to_numeric(df["DX2"], errors="coerce")
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")

    # 그룹 단위 계산
    rows = []
    for rid, g in df.groupby("RID", dropna=True):
        if pd.isna(rid):
            continue

        bl = pick_baseline(g)
        m6 = pick_near_6m(g)
        if m6 is None:
            continue

        # baseline DX2를 대표값으로 사용(없으면 m6에서 사용)
        dx2 = bl.get("DX2", np.nan)
        if pd.isna(dx2):
            dx2 = m6.get("DX2", np.nan)

        # 값들
        icv = pd.to_numeric(bl["ICV"], errors="coerce")
        v_bl = pd.to_numeric(bl["Ventricles_bl"], errors="coerce")
        v_6 = pd.to_numeric(m6["Ventricles"], errors="coerce")
        m_6 = pd.to_numeric(m6["Month_bl"], errors="coerce")

        if pd.isna(icv) or icv == 0 or pd.isna(v_bl) or pd.isna(v_6) or pd.isna(m_6):
            continue

        v_bl_norm = v_bl / icv
        v_6_norm = v_6 / icv
        delta_6m = v_6_norm - v_bl_norm
        slope_6m = delta_6m / TARGET_M

        rows.append({
            "RID": int(rid),
            "DX2": int(dx2) if not pd.isna(dx2) else np.nan,
            "Month_bl_matched": float(m_6),
            "V_bl_norm": float(v_bl_norm),
            "V_6m_norm": float(v_6_norm),
            "delta_6m": float(delta_6m),
            "slope_6m": float(slope_6m),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[Wrote] {OUT_CSV} (n={len(out)})")

    # ---- plotting helper ----
    def scatter(sub_df: pd.DataFrame, tag: str, title: str):
        plt.figure()
        x = sub_df["Month_bl_matched"]
        y = sub_df["slope_6m"]
        plt.scatter(x, y, alpha=0.35, s=10, c=COLORS[tag])
        plt.xlabel("Matched Month_bl (closest to 6 months)")
        plt.ylabel("Ventricles slope_6m = Δ(V/ICV)/6")
        plt.title(title)
        out_png = OUT_DIR / f"fixed6m_scatter_{tag}_01_22_26.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_png}")

    # ALL
    scatter(out.dropna(subset=["slope_6m", "Month_bl_matched"]), "ALL",
            f"Fixed-window (6m±{TOL_M:.0f}) Ventricles slope: ALL")

    # Per DX2
    for dx, name in LABELS.items():
        sub = out[out["DX2"] == dx].dropna(subset=["slope_6m", "Month_bl_matched"])
        scatter(sub, name, f"Fixed-window (6m±{TOL_M:.0f}) Ventricles slope: {name}")

if __name__ == "__main__":
    main()
