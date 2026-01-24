#01-20-2026 Roy Seo & ChatGPT
#돌리는 방법:
#python train_1_20_26.py --config ./configs/baseline/baseline_v1_tab10_1_19_26.yaml

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml

from src.data.adni_dataset_1_19_26 import ADNIDataset
from src.models.baseline_cnn_tabular_1_19_26 import Baseline3DCNNTabular_1_19_26


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic-ish behavior (can reduce speed a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Git metadata
# =========================
def _run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        return None


def get_git_metadata():
    git_hash = _run_cmd(["git", "rev-parse", "HEAD"])
    git_hash_short = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
    dirty = _run_cmd(["git", "status", "--porcelain"])
    return {
        "git_hash": git_hash,
        "git_hash_short": git_hash_short,
        "git_dirty": (dirty is not None and len(dirty) > 0),
    }


# =========================
# Metrics
# =========================
@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor, num_classes: int):
    pred = torch.argmax(logits, dim=1)
    acc = (pred == y).float().mean().item()

    # balanced acc
    ba_parts = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            ba_parts.append((pred[mask] == y[mask]).float().mean().item())
    balanced_acc = float(sum(ba_parts) / max(len(ba_parts), 1))
    return {"acc": float(acc), "balanced_acc": float(balanced_acc)}


def run_one_epoch(model, loader, device, optimizer=None, scaler=None, num_classes=3):
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)


    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_y = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["tabular"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(img, tab)
            loss = ce(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    metrics = compute_metrics(all_logits, all_y, num_classes=num_classes)
    metrics["loss"] = float(total_loss / max(n_batches, 1))
    return metrics

@torch.no_grad()
def dump_predictions(model, loader, device, num_classes: int, out_csv: Path, split_name: str):
    model.eval()

    rows = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tab = batch["tabular"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits = model(img, tab)                    # [B, C]
        probs = F.softmax(logits, dim=1)            # [B, C]
        pred = torch.argmax(logits, dim=1)          # [B]

        logits = logits.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        # ---- metadata (있으면 저장, 없으면 NaN/None) ----
        # ADNIDataset이 아래 key들을 반환하도록 해두면 가장 좋음:
        # "RID", "Image Data ID", "Month_bl", "DX2", "EXAMDATE"
        rid = batch.get("RID", batch.get("rid", None))
        img_id = batch.get("Image Data ID", batch.get("image_id", None))
        month_bl = batch.get("Month_bl", batch.get("month_bl", None))
        dx2 = batch.get("DX2", batch.get("dx2", None))
        examdate = batch.get("EXAMDATE", batch.get("examdate", None))

        # tensor면 cpu로
        def to_list(x):
            if x is None:
                return [None] * len(y)
            if torch.is_tensor(x):
                return x.detach().cpu().numpy().tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x] * len(y)

        rid_l = to_list(rid)
        img_id_l = to_list(img_id)
        month_bl_l = to_list(month_bl)
        dx2_l = to_list(dx2)
        examdate_l = to_list(examdate)

        for i in range(len(y)):
            r = {
                "split": split_name,
                "y_true": int(y[i]),
                "y_pred": int(pred[i]),
            }
            # meta
            r["RID"] = rid_l[i]
            r["Image Data ID"] = img_id_l[i]
            r["Month_bl"] = month_bl_l[i]
            r["DX2"] = dx2_l[i]
            r["EXAMDATE"] = examdate_l[i]

            # logits/probs
            for c in range(num_classes):
                r[f"logit_c{c}"] = float(logits[i, c])
                r[f"prob_c{c}"] = float(probs[i, c])

            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[Saved preds] {out_csv} (n={len(df)})")



# =========================
# Run directory helper
# =========================
def make_run_dir(output_dir: str, exp_name: str, seed: int, date_tag: str):
    """
    Create a unique run directory:
      <output_dir>/<exp_name>/<exp_name>_<date_tag>_<timestamp>_seed<seed>_<gitshort>/
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    git_meta = get_git_metadata()
    gh = git_meta.get("git_hash_short") or "nogit"

    base = Path(output_dir) / exp_name
    base.mkdir(parents=True, exist_ok=True)

    run_name = f"{exp_name}_{date_tag}_{ts}_seed{seed}_{gh}"
    run_dir = base / run_name

    # prevent accidental overwrite
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name, ts, git_meta


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    # (1) load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    # (2) extract experiment fields
    exp_name = exp["name"]
    seed = int(exp.get("seed", 0))

    # Force date_tag for today if you want; otherwise reads from config.
    # You asked: "output files name에도 1_20_26 붙일 수 있으면 붙여줘"
    # So we default to config's date_tag, and if missing, use "1_20_26".
    date_tag = datetime.now().strftime("%m_%d_%y")

    set_seed(seed)

    # (3) create run_dir
    run_dir, run_name, ts, git_meta = make_run_dir(
        output_dir=exp["output_dir"],
        exp_name=exp_name,
        seed=seed,
        date_tag=date_tag,
    )

    # (4) save config snapshot + split copy
    config_snap_path = run_dir / f"config_{date_tag}.yaml"
    shutil.copy2(args.config, config_snap_path)

    split_json = dcfg.get("split_json", None)
    if split_json and os.path.exists(split_json):
        shutil.copy2(split_json, run_dir / f"split_{date_tag}.json")

    # (5) save meta
    meta = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "timestamp": ts,
        "date_tag": date_tag,
        "seed": seed,
        "command": " ".join(os.sys.argv),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **git_meta,
    }
    meta_path = run_dir / f"meta_{date_tag}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # (6) device
    device_req = str(tcfg.get("device", "cuda"))
    device = torch.device(device_req if (device_req.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # (7) datasets
    def make_ds(split_name: str):
        return ADNIDataset(
            csv_path=dcfg["csv_path"],
            split_json=dcfg["split_json"],
            split=split_name,
            image_id_col=dcfg["image_id_col"],
            image_dir=dcfg["image_dir"],
            image_glob_pattern=dcfg["image_glob_pattern"],
            rid_col=dcfg["rid_col"],
            label_col=dcfg["label_col"],
            tabular_cols=dcfg.get("tabular_cols", None),   # legacy
            cont_cols=dcfg.get("cont_cols", None),         # new
            cat_cols=dcfg.get("cat_cols", None),           # new
            tabular_standardize=bool(dcfg.get("tabular_standardize", True)),
            sex_encoding=dcfg.get("sex_encoding", "binary"),
        )

    train_ds = make_ds("train")
    val_ds = make_ds("val")
    test_ds = make_ds("test")

    bs = int(tcfg["batch_size"])
    nw = int(tcfg.get("num_workers", 2))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # (8) tabular dim
    tab_cols = dcfg.get("tabular_cols", None)
    cont_cols = dcfg.get("cont_cols", []) or []
    cat_cols = dcfg.get("cat_cols", []) or []
    if tab_cols is not None:
        tab_dim = len(tab_cols)
    else:
        tab_dim = len(cont_cols) + len(cat_cols)

    num_classes = int(mcfg["num_classes"])

    # (9) model
    model = Baseline3DCNNTabular_1_19_26(tabular_dim=tab_dim, num_classes=num_classes).to(device)

    # (10) optimizer + amp
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(tcfg.get("amp", True)) and device.type == "cuda"
    )

    # (11) output paths (all include date_tag)
    ckpt_best_path = run_dir / f"model_best_{date_tag}.pt"
    ckpt_final_path = run_dir / f"model_final_{date_tag}.pt"
    ckpt_last_path = run_dir / f"checkpoint_last_{date_tag}.pt"

    metrics_json_path = run_dir / f"metrics_{date_tag}.json"
    metrics_epoch_csv_path = run_dir / f"metrics_epoch_{date_tag}.csv"

    # (12) training loop
    epochs = int(tcfg["epochs"])
    best_val = float("inf")
    best_epoch = -1

    rows = []
    for ep in range(1, epochs + 1):
        tr = run_one_epoch(model, train_loader, device, optimizer=opt, scaler=scaler, num_classes=num_classes)
        va = run_one_epoch(model, val_loader, device, optimizer=None, scaler=None, num_classes=num_classes)

        row = {
            "epoch": ep,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_balanced_acc": tr["balanced_acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_balanced_acc": va["balanced_acc"],
        }
        rows.append(row)

        print(
            f"[{ep}/{epochs}] "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            f"val loss={va['loss']:.4f} acc={va['acc']:.4f}"
        )

        # always save "last"
        torch.save({"model_state": model.state_dict(), "epoch": ep}, ckpt_last_path)

        # save best by val_loss
        if va["loss"] < best_val:
            best_val = float(va["loss"])
            best_epoch = ep
            torch.save({"model_state": model.state_dict(), "epoch": ep, "val_loss": best_val}, ckpt_best_path)

    # final checkpoint (end of training)
    torch.save({"model_state": model.state_dict(), "epoch": epochs}, ckpt_final_path)

    # (13) test with best checkpoint
    best = torch.load(ckpt_best_path, map_location=device)
    model.load_state_dict(best["model_state"])
    te = run_one_epoch(model, test_loader, device, optimizer=None, scaler=None, num_classes=num_classes)

    preds_train_csv = run_dir / f"preds_train_{date_tag}.csv"
    preds_val_csv   = run_dir / f"preds_val_{date_tag}.csv"
    preds_test_csv  = run_dir / f"preds_test_{date_tag}.csv"

    dump_predictions(model, train_loader, device, num_classes, preds_train_csv, "train")
    dump_predictions(model, val_loader, device, num_classes, preds_val_csv, "val")
    dump_predictions(model, test_loader, device, num_classes, preds_test_csv, "test")


    # (14) metrics.json (single official file)
    metrics_all = {
        "seed": seed,
        "date_tag": date_tag,
        "best": {
            "epoch": int(best.get("epoch", best_epoch)),
            "val_loss": float(best.get("val_loss", best_val)),
        },
        "test": {
            "loss": float(te["loss"]),
            "acc": float(te["acc"]),
            "balanced_acc": float(te["balanced_acc"]),
        },
        "train_summary": {
            "epochs": epochs,
            "batch_size": bs,
            "lr": float(tcfg["lr"]),
            "weight_decay": float(tcfg.get("weight_decay", 0.0)),
            "amp": bool(tcfg.get("amp", True)) and device.type == "cuda",
            "device": str(device),
        },
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_all, f, indent=2)

    # (15) metrics_epoch CSV
    if len(rows) > 0:
        fieldnames = list(rows[0].keys())
        with open(metrics_epoch_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print("\n[Saved run]")
    print("  run_dir:", str(run_dir))
    print("  metrics:", str(metrics_json_path))
    print("  epoch_csv:", str(metrics_epoch_csv_path))
    print("  best_ckpt:", str(ckpt_best_path))
    print("  last_ckpt:", str(ckpt_last_path))
    print("  final_ckpt:", str(ckpt_final_path))
    print("\n[Best/Test]")
    print("  best_epoch:", metrics_all["best"]["epoch"])
    print("  test_acc:", metrics_all["test"]["acc"])
    print("  test_balanced_acc:", metrics_all["test"]["balanced_acc"])


if __name__ == "__main__":
    main()