#!/usr/bin/env python3
import argparse
import glob
import json
import os
from datetime import datetime
from statistics import mean, pstdev

import pandas as pd


DEFAULT_KEYS = [
    "best.epoch",
    "test.acc",
    "test.balanced_acc",
    # 필요하면 아래도 추가 가능:
    # "val_best_acc",
    # "val_best_loss",
]

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def find_seed_runs(runs_dir: str):
    """
    Find newest run_dir per seed under runs_dir.
    We assume run folder name contains 'seed0', 'seed1', 'seed2'.
    """
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*seed*")), reverse=True)
    seeds = {}
    for d in run_dirs:
        base = os.path.basename(d)
        for s in ["seed0", "seed1", "seed2"]:
            if s in base and s not in seeds:
                seeds[s] = d
    return seeds

def find_metrics_json(run_dir: str):
    """
    Your run saves metrics as metrics_01_20_26.json (date suffix may vary).
    So just pick the newest metrics_*.json inside run_dir.
    """
    cand = sorted(glob.glob(os.path.join(run_dir, "metrics_*.json")))
    if not cand:
        return None
    return cand[-1]  # newest by name; ok because date suffix embedded

def get_by_path(d, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def extract_metrics(d: dict, keys):
    out = {}
    for k in keys:
        out[k] = get_by_path(d, k)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True, help="e.g., runs/baseline_v1_tab10")
    ap.add_argument("--out_dir", required=True, help="e.g., runs/baseline_v1_tab10/_summary")
    ap.add_argument("--keys", nargs="*", default=DEFAULT_KEYS, help="metric keys to aggregate")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seed_runs = find_seed_runs(args.runs_dir)
    missing = [s for s in ["seed0", "seed1", "seed2"] if s not in seed_runs]
    if missing:
        raise SystemExit(f"[ERROR] Missing run dirs for: {missing}. Found: {seed_runs}")

    rows = []
    raw = {}
    for seed, run_dir in seed_runs.items():
        mpath = find_metrics_json(run_dir)
        if mpath is None:
            raise SystemExit(f"[ERROR] No metrics_*.json found in {run_dir}")
        md = load_json(mpath)
        ex = extract_metrics(md, args.keys)
        raw[seed] = {
            "run_dir": run_dir,
            "metrics_path": mpath,
            **ex
        }
        rows.append((seed, ex))

    # Build table: metric x seeds + mean + std
    table = []
    for k in args.keys:
        vals = []
        per_seed = {}
        for seed, ex in rows:
            v = ex.get(k, None)
            per_seed[seed] = v
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            mu = mean(vals)
            sd = pstdev(vals) if len(vals) > 1 else 0.0
        else:
            mu, sd = None, None

        table.append({
            "metric": k,
            "seed0": per_seed.get("seed0"),
            "seed1": per_seed.get("seed1"),
            "seed2": per_seed.get("seed2"),
            "mean": mu,
            "std": sd,
        })

    df = pd.DataFrame(table)

    date_tag = None
    for seed in ["seed0", "seed1", "seed2"]:
        if seed in raw and "date_tag" in raw[seed]:
            date_tag = raw[seed]["date_tag"]
            break
    if date_tag is None:        
        date_tag = datetime.now().strftime("%m_%d_%y")

    out_csv = os.path.join(args.out_dir, f"aggregate_seed_metrics_{date_tag}.csv")
    df.to_csv(out_csv, index=False)

    out_json = os.path.join(args.out_dir, f"aggregate_seed_metrics_{date_tag}.json")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs_dir": args.runs_dir,
        "keys": args.keys,
        "per_seed": raw,
        "aggregate": df.to_dict(orient="records"),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n[Wrote]")
    print(" ", out_csv)
    print(" ", out_json)

    # Pretty print 핵심 몇 개
    print("\n[Summary]")
    for r in table:
        if r["mean"] is None:
            continue
        print(f"  {r['metric']}: mean={r['mean']:.6f}  std={r['std']:.6f}  (seed0={r['seed0']}, seed1={r['seed1']}, seed2={r['seed2']})")

if __name__ == "__main__":
    main()