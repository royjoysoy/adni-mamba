"""
ChatGPT + Roy Seo 01-22-2026

[한국어]
Ventricles (ICV normalized) 진행 속도를 "구간별(stratified)"로 분석합니다.
출력은 ALL / CN / MCI / AD 4개 그룹 각각에 대해:
- 6개월 bin(0-6, 6-12, 12-18, 18-24)별 점들을 색으로 구분한 scatter plot 생성
- 모든 플롯에서 x/y 축 범위를 동일하게 고정하여 비교 가능하게 함

중요:
- Month_bl(개인 baseline 이후 경과개월) 기준 분석입니다. 달력 연도(2005/2006) 변화 분석은 별도입니다.

두 가지 slope를 모두 생성:
(A) cumulative slope: baseline -> 해당 visit (기존 방식)
(B) interval slope: bin 경계 근처 두 visit 사이 변화율 (구간 변화 감지에 더 적합)

[English]
Interval-stratified Ventricles progression analysis with common axis limits across ALL/CN/MCI/AD.
Generates both cumulative slopes and interval slopes.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = "/home/ads_sry/royseo/projects/adni-mamba/data_0/ADNI_master_merged_12-17-2025.csv"
OUT_DIR = Path("/home/ads_sry/royseo/projects/adni-mamba/A_1_19_26/reports/week2/day1_roi_progression")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- settings -----
# 6-month bins up to 24 months
BINS = [(0,6), (6,12), (12,18), (18,24)]
BIN_LABELS = [f"{a}-{b}m" for a,b in BINS]

# match tolerance around bin boundaries (guesses like 6±2m)
TOL = 2.0

# group labels/colors (consistent across all plots)
DX_LABEL = {0:"CN", 1:"MCI", 2:"AD"}
BIN_COLOR = {
    "0-6m":   "tab:blue",
    "6-12m":  "tab:orange",
    "12-18m": "tab:green",
    "18-24m": "tab:red",
}

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def pick_near_month(g: pd.DataFrame, target: float, tol: float):
    """Pick the single row with Month_bl closest to target within ±tol, excluding Month_bl==0 unless target==0."""
    gg = g.copy()
    gg["Month_bl"] = to_num(gg["Month_bl"])
    gg = gg.dropna(subset=["Month_bl"])
    cand = gg[(gg["Month_bl"] >= target - tol) & (gg["Month_bl"] <= target + tol)]
    if target > 0:
        cand = cand[cand["Month_bl"] > 0]
    if len(cand) == 0:
        return None
    cand = cand.assign(_dist=(cand["Month_bl"] - target).abs())
    return cand.sort_values(["_dist","Month_bl"]).iloc[0]

def compute_cumulative(df):
    """Compute cumulative slope at each visit: (V/ICV - V_bl/ICV)/Month_bl."""
    out = df.copy()
    out["Month_bl"] = to_num(out["Month_bl"])
    out["DX2"] = to_num(out["DX2"])
    out["ICV"] = to_num(out["ICV"])
    out["Ventricles"] = to_num(out["Ventricles"])
    out["Ventricles_bl"] = to_num(out["Ventricles_bl"])

    # normalized values
    out["V_norm"] = out["Ventricles"] / out["ICV"]
    out["V_bl_norm"] = out["Ventricles_bl"] / out["ICV"]

    out["delta_bl"] = out["V_norm"] - out["V_bl_norm"]
    out["slope_cum"] = out["delta_bl"] / out["Month_bl"]

    # exclude baseline / invalid
    out.loc[out["Month_bl"] <= 0, "slope_cum"] = np.nan
    return out

def assign_bin(month):
    for a,b in BINS:
        if month > a and month <= b:
            return f"{a}-{b}m"
    return None

def compute_interval(df):
    """
    For each RID and each bin (a,b), compute interval slope between near-a and near-b visits:
    slope_int = (V_norm(b) - V_norm(a)) / (Month_b - Month_a)
    Using tolerance around boundaries.
    """
    rows = []
    df2 = df.copy()
    df2["RID"] = to_num(df2["RID"])
    df2["DX2"] = to_num(df2["DX2"])
    df2["Month_bl"] = to_num(df2["Month_bl"])
    df2["ICV"] = to_num(df2["ICV"])
    df2["Ventricles"] = to_num(df2["Ventricles"])
    df2["Ventricles_bl"] = to_num(df2["Ventricles_bl"])

    for rid, g in df2.groupby("RID", dropna=True):
        if pd.isna(rid):
            continue

        # need valid ICV and baseline norm (from Ventricles_bl / ICV) to define consistent normalization
        # We'll use each row's own Ventricles/ICV; ICV should be stable, but we take ICV from the chosen row.
        # DX2: use baseline if exists else first non-null
        dx2 = g.loc[g["DX2"].notna(), "DX2"].iloc[0] if g["DX2"].notna().any() else np.nan

        for a,b in BINS:
            ra = pick_near_month(g, a if a>0 else 0.0, TOL)
            rb = pick_near_month(g, b, TOL)
            if ra is None or rb is None:
                continue

            ma = float(ra["Month_bl"])
            mb = float(rb["Month_bl"])
            if not (mb > ma):
                continue

            icv_a = ra["ICV"]
            icv_b = rb["ICV"]
            va = ra["Ventricles"]
            vb = rb["Ventricles"]
            if pd.isna(icv_a) or pd.isna(icv_b) or icv_a==0 or icv_b==0 or pd.isna(va) or pd.isna(vb):
                continue

            vna = float(va / icv_a)
            vnb = float(vb / icv_b)
            slope_int = (vnb - vna) / (mb - ma)

            rows.append({
                "RID": int(rid),
                "DX2": int(dx2) if not pd.isna(dx2) else np.nan,
                "bin": f"{a}-{b}m",
                "Month_mid": (ma + mb) / 2.0,
                "Month_a": ma,
                "Month_b": mb,
                "slope_int": float(slope_int),
            })

    return pd.DataFrame(rows)

def common_limits(values_list, pad_frac=0.05):
    """Compute common y-limits across multiple series with small padding."""
    vals = np.concatenate([v[np.isfinite(v)] for v in values_list if len(v)>0], axis=0) if values_list else np.array([])
    if vals.size == 0:
        return (-1, 1)
    lo, hi = np.percentile(vals, [1, 99])  # robust to outliers
    if lo == hi:
        lo, hi = lo - 1e-4, hi + 1e-4
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)

def plot_stratified_scatter(df_plot, ycol, title_prefix, out_png):
    """
    Scatter: x = Month_bl (or Month_mid), y = slope, colored by bins.
    Axis limits are set later globally.
    """
    plt.figure(figsize=(7,5))
    for binlab in BIN_LABELS:
        sub = df_plot[df_plot["bin"] == binlab]
        if len(sub) == 0:
            continue
        plt.scatter(sub["x"], sub[ycol], alpha=0.35, s=12, c=BIN_COLOR[binlab], label=binlab)
    plt.xlabel(df_plot.attrs.get("xlabel", "Month_bl"))
    plt.ylabel(df_plot.attrs.get("ylabel", ycol))
    plt.title(title_prefix)
    plt.legend(frameon=False)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_png}")

def main():
    df = pd.read_csv(IN_CSV, low_memory=False)

    # ---- cumulative ----
    cum = compute_cumulative(df)
    cum["bin"] = cum["Month_bl"].apply(lambda m: assign_bin(m) if pd.notna(m) else None)
    cum = cum[cum["bin"].notna()].copy()
    cum = cum.rename(columns={"Month_bl":"x"})
    cum.attrs["xlabel"] = "Month_bl (months since baseline)"
    cum.attrs["ylabel"] = "Cumulative slope = Δ(V/ICV)/Month_bl"

    # ---- interval ----
    inter = compute_interval(df)
    if len(inter) > 0:
        inter = inter.rename(columns={"Month_mid":"x"})
        inter.attrs["xlabel"] = "Midpoint months of interval"
        inter.attrs["ylabel"] = "Interval slope = Δ(V/ICV)/ΔMonth"

    # Prepare group subsets for plotting
    groups = [("ALL", None), ("CN", 0), ("MCI", 1), ("AD", 2)]

    # Collect y-values to set common y-limits across ALL/CN/MCI/AD (separately for cum and interval)
    cum_vals = []
    inter_vals = []

    cum_subs = {}
    inter_subs = {}

    for name, dx in groups:
        if dx is None:
            csub = cum.dropna(subset=["slope_cum","x"])
        else:
            csub = cum[(cum["DX2"] == dx)].dropna(subset=["slope_cum","x"])
        cum_subs[name] = csub
        cum_vals.append(csub["slope_cum"].to_numpy())

        if len(inter) > 0:
            if dx is None:
                isub = inter.dropna(subset=["slope_int","x"])
            else:
                isub = inter[(inter["DX2"] == dx)].dropna(subset=["slope_int","x"])
            inter_subs[name] = isub
            inter_vals.append(isub["slope_int"].to_numpy())

    ylo_c, yhi_c = common_limits(cum_vals)
    ylo_i, yhi_i = common_limits(inter_vals) if len(inter_vals)>0 else (None,None)

    # ---- plot cumulative (common y across groups) ----
    for name, _ in groups:
        sub = cum_subs[name].copy()
        sub.attrs = cum.attrs
        out_png = OUT_DIR / f"strat6m_cumulative_Ventricles_{name}_01_22_26.png"
        plt.figure(figsize=(7,5))
        for binlab in BIN_LABELS:
            ss = sub[sub["bin"] == binlab]
            if len(ss)==0:
                continue
            plt.scatter(ss["x"], ss["slope_cum"], alpha=0.35, s=12, c=BIN_COLOR[binlab], label=binlab)
        plt.xlabel(sub.attrs["xlabel"])
        plt.ylabel(sub.attrs["ylabel"])
        plt.title(f"Cumulative Ventricles slope by interval bins: {name}")
        plt.ylim(ylo_c, yhi_c)
        plt.xlim(0, 24)
        plt.legend(frameon=False)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_png}")

    # ---- plot interval slopes (common y across groups) ----
    if len(inter) > 0:
        for name, _ in groups:
            sub = inter_subs[name].copy()
            sub.attrs = inter.attrs
            out_png = OUT_DIR / f"strat6m_interval_Ventricles_{name}_01_22_26.png"
            plt.figure(figsize=(7,5))
            for binlab in BIN_LABELS:
                ss = sub[sub["bin"] == binlab]
                if len(ss)==0:
                    continue
                plt.scatter(ss["x"], ss["slope_int"], alpha=0.35, s=12, c=BIN_COLOR[binlab], label=binlab)
            plt.xlabel(sub.attrs["xlabel"])
            plt.ylabel(sub.attrs["ylabel"])
            plt.title(f"Interval Ventricles slope by bins: {name} (boundary match ±{TOL}m)")
            plt.ylim(ylo_i, yhi_i)
            plt.xlim(0, 24)
            plt.legend(frameon=False)
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[Saved] {out_png}")

    # Save derived tables
    cum_out = OUT_DIR / "strat6m_cumulative_ventricles_rows_01_22_26.csv"
    cum_sub = cum[["RID","DX2","x","bin","slope_cum"]].copy()
    cum_sub.to_csv(cum_out, index=False)
    print(f"[Wrote] {cum_out} (n={len(cum_sub)})")

    if len(inter) > 0:
        int_out = OUT_DIR / "strat6m_interval_ventricles_rows_01_22_26.csv"
        inter.to_csv(int_out, index=False)
        print(f"[Wrote] {int_out} (n={len(inter)})")

if __name__ == "__main__":
    main()
