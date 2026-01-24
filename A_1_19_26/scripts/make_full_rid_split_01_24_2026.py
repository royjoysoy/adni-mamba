# scripts/make_full_rid_split_01_24_2026.py
"""
Create RID-level train/val/test split for ADNI (longitudinal-safe).

- All scans from the same RID go to the same split
- Stratified by label_col (DX2 recommended)
- Outputs JSON with train_rids / val_rids / test_rids
"""

import argparse
import json
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rid_col", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.rid_col, args.label_col])

    # only keep valid labels (0/1/2)
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    df = df[df[args.label_col].isin([0, 1, 2])]

    # one label per RID (majority vote)
    rid_label = (
        df.groupby(args.rid_col)[args.label_col]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    rng = np.random.RandomState(args.seed)

    train_rids, val_rids, test_rids = [], [], []

    for label in sorted(rid_label[args.label_col].unique()):
        rids = rid_label[rid_label[args.label_col] == label][args.rid_col].values
        rng.shuffle(rids)

        n = len(rids)
        n_train = int(n * args.train)
        n_val = int(n * args.val)

        train_rids.extend(rids[:n_train].tolist())
        val_rids.extend(rids[n_train:n_train + n_val].tolist())
        test_rids.extend(rids[n_train + n_val:].tolist())

    out = {
        "train_rids": sorted(train_rids),
        "val_rids": sorted(val_rids),
        "test_rids": sorted(test_rids),
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # summary
    print("Saved:", args.out)
    print("RIDs:", {k: len(v) for k, v in out.items()})

    # scan-level counts
    for split, rids in out.items():
        n_scans = df[df[args.rid_col].isin(rids)].shape[0]
        print(f"{split}: scans={n_scans}")


if __name__ == "__main__":
    main()