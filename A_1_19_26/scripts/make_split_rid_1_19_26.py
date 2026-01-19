# scripts/make_split_rid.py
import argparse, json, os, random
import pandas as pd
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rid_col", default="RID_x")
    ap.add_argument("--label_col", default="DX_bl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.rid_col, args.label_col]).copy()

    # subject-level table: one label per RID (DX_bl should already be stable)
    subj = (
        df[[args.rid_col, args.label_col]]
        .drop_duplicates()
        .rename(columns={args.rid_col: "RID", args.label_col: "LABEL"})
    )

    # stratified split by LABEL at RID level
    rng = random.Random(args.seed)
    rids_by_label = {}
    for lab, g in subj.groupby("LABEL"):
        rids = g["RID"].tolist()
        rng.shuffle(rids)
        rids_by_label[lab] = rids

    train_rids, val_rids, test_rids = [], [], []
    for lab, rids in rids_by_label.items():
        n = len(rids)
        n_train = int(round(n * args.train))
        n_val = int(round(n * args.val))
        # remainder goes to test
        tr = rids[:n_train]
        va = rids[n_train:n_train + n_val]
        te = rids[n_train + n_val:]
        train_rids += tr
        val_rids += va
        test_rids += te

    # shuffle final lists to avoid label blocks
    rng.shuffle(train_rids); rng.shuffle(val_rids); rng.shuffle(test_rids)

    split = {
        "version": "adni_rid_split_v1",
        "seed": args.seed,
        "fractions": {"train": args.train, "val": args.val, "test": args.test},
        "rid_col": args.rid_col,
        "label_col": args.label_col,
        "train_rids": train_rids,
        "val_rids": val_rids,
        "test_rids": test_rids,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)

    # quick summary
    def summarize(rids):
        d = df[df[args.rid_col].isin(rids)]
        rid_n = d[args.rid_col].nunique()
        img_n = len(d)
        lab = subj[subj["RID"].isin(rids)]["LABEL"].tolist()
        return {"n_rid": rid_n, "n_images": img_n, "label_counts": dict(Counter(lab))}

    print("Saved:", args.out)
    print("TRAIN:", summarize(train_rids))
    print("VAL  :", summarize(val_rids))
    print("TEST :", summarize(test_rids))

if __name__ == "__main__":
    main()