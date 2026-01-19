import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from src.data.adni_dataset_1_19_26 import ADNIDataset
from src.models.baseline_cnn_tabular_1_19_26 import Baseline3DCNNTabular_1_19_26


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_metadata():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            return None

    git_hash = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return {
        "git_hash": git_hash,
        "git_dirty": (dirty is not None and len(dirty) > 0),
    }


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor, num_classes: int):
    pred = torch.argmax(logits, dim=1)
    acc = (pred == y).float().mean().item()

    ba_parts = []
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            ba_parts.append((pred[mask] == y[mask]).float().mean().item())
    balanced_acc = float(sum(ba_parts) / max(len(ba_parts), 1))
    return {"acc": acc, "balanced_acc": balanced_acc}


def run_one_epoch(model, loader, device, optimizer=None, scaler=None, num_classes=3):
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_y = []

    for batch in loader:
        img = batch["image"].to(device)
        tab = batch["tabular"].to(device)
        y = batch["label"].to(device)

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

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    metrics = compute_metrics(all_logits, all_y, num_classes=num_classes)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    seed = int(exp.get("seed", 0))
    set_seed(seed)

    date_tag = exp.get("date_tag", datetime.now().strftime("%m_%d_%y"))
    run_name = f"{exp['name']}_seed{seed}_{date_tag}"
    run_dir = os.path.join(exp["output_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    # save config + split copy
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))
    split_json = cfg["data"]["split_json"]
    if os.path.exists(split_json):
        shutil.copy2(split_json, os.path.join(run_dir, "split.json"))

    # metadata
    meta = {
        "run_name": run_name,
        "command": f"python train_1_19_26.py --config {args.config}",
        "seed": seed,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **get_git_metadata(),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    dcfg = cfg["data"]
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    device = torch.device(tcfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # datasets
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

    tab_cols = dcfg.get("tabular_cols", None)
    cont_cols = dcfg.get("cont_cols", []) or []
    cat_cols = dcfg.get("cat_cols", []) or []

    if tab_cols is not None:
        tab_dim = len(tab_cols)
    else:
        tab_dim = len(cont_cols) + len(cat_cols)

    num_classes = int(mcfg["num_classes"])
    model = Baseline3DCNNTabular_1_19_26(tabular_dim=tab_dim, num_classes=num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(tcfg.get("amp", True)) and device.type == "cuda")

    best_val = float("inf")
    best_path = os.path.join(run_dir, "model_best.pt")
    final_path = os.path.join(run_dir, "model_final.pt")

    epochs = int(tcfg["epochs"])
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

        print(f"[{ep}/{epochs}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | val loss={va['loss']:.4f} acc={va['acc']:.4f}")

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"model_state": model.state_dict(), "epoch": ep, "val_loss": best_val}, best_path)

    torch.save({"model_state": model.state_dict(), "epoch": epochs}, final_path)

    # test best checkpoint
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state"])
    te = run_one_epoch(model, test_loader, device, optimizer=None, scaler=None, num_classes=num_classes)

    metrics_best = {
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "test_loss": float(te["loss"]),
        "test_acc": float(te["acc"]),
        "test_balanced_acc": float(te["balanced_acc"]),
    }
    with open(os.path.join(run_dir, "metrics_best.json"), "w") as f:
        json.dump(metrics_best, f, indent=2)

    # write csv
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        cols = list(rows[0].keys())
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print("Saved run to:", run_dir)
    print("Best metrics:", metrics_best)


if __name__ == "__main__":
    main()
