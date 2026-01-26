# 01-26-2026 Roy Seo & ChatGPT
# ------------------------------------------------------------
# File: train_cae_transfer_01_26_2026.py
#
# Purpose:
#   Supervised 3-way classification (CN/MCI/AD) using a pretrained CAE encoder (CAE3D).
#   - Linear Probe: freeze_encoder=True  (train head only)
#   - Fine-tune:    freeze_encoder=False (train encoder + head)
#
# Expected image shape:
#   (B, 1, 64, 64, 64)
#
# Run (Linear Probe):
#   python train_cae_transfer_01_26_2026.py --config ./configs/transfer/cae_linear_probe_01_26_2026.yaml
#
# Run (Fine-tune):
#   python train_cae_transfer_01_26_2026.py --config ./configs/transfer/cae_finetune_01_26_2026.yaml
#
# Outputs:
#   runs/<exp_name>/<run_name>/
#     - config_<date>.yaml (snapshot)
#     - meta_<date>.json
#     - model_best_<date>.pt / model_final_<date>.pt
#     - metrics_<date>.json / metrics_epoch_<date>.csv
# ------------------------------------------------------------

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.adni_dataset_1_19_26 import ADNIDataset
from src.models_rs.cae_3d_01_24_2026 import CAE3D


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Git metadata
# -------------------------
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


def make_run_dir(output_dir: str, exp_name: str, seed: int, date_tag: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    git_meta = get_git_metadata()
    gshort = git_meta.get("git_hash_short") or "nogit"
    run_name = f"{exp_name}_{date_tag}_{ts}_seed{seed}_{gshort}"
    run_dir = output_dir / exp_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name, ts, git_meta


# -------------------------
# Metrics
# -------------------------
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
    return {"acc": float(acc), "balanced_acc": float(balanced_acc)}


# -------------------------
# Model: CAE encoder + head
# -------------------------
class CAEClassifier(nn.Module):
    """
    Use CAE3D encoder blocks (enc1/enc2/enc3) as feature extractor.
    Input:  (B,1,64,64,64)
    After pools: 64->32->16->8
    z: (B,128,8,8,8) -> GAP -> (B,128) -> head -> logits
    """
    def __init__(self, in_ch: int, num_classes: int, head: str = "linear", head_hidden: int = 256):
        super().__init__()
        cae = CAE3D(in_ch=in_ch)
        self.enc1 = cae.enc1
        self.enc2 = cae.enc2
        self.enc3 = cae.enc3

        self.pool = nn.AdaptiveAvgPool3d(1)  # (B,C,1,1,1)
        feat_dim = 128

        head = head.lower()
        if head == "linear":
            self.head = nn.Linear(feat_dim, num_classes)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            raise ValueError(f"Unknown head='{head}'. Use 'linear' or 'mlp'.")

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor = None) -> torch.Tensor:
        # tabular ignored (kept for dataset compatibility)
        z = self.enc3(self.enc2(self.enc1(x_img)))
        z = self.pool(z).flatten(1)
        return self.head(z)


def _extract_state_dict(obj: object) -> dict:
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    raise ValueError("Checkpoint is not a dict-like object.")


def load_cae_encoder_weights(model: CAEClassifier, ckpt_path: str, in_ch: int):
    """
    Load CAE checkpoint weights into classifier encoder.
    Only copy enc1/enc2/enc3 matching keys (ignores decoder).
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(obj)

    tmp = CAE3D(in_ch=in_ch)
    tmp.load_state_dict(state, strict=False)
    cae_state = tmp.state_dict()

    cls_state = model.state_dict()
    copied = 0
    for k in cls_state.keys():
        if k.startswith(("enc1.", "enc2.", "enc3.")) and k in cae_state:
            if cae_state[k].shape == cls_state[k].shape:
                cls_state[k] = cae_state[k]
                copied += 1
    model.load_state_dict(cls_state, strict=False)
    return copied


def set_encoder_trainable(model: CAEClassifier, trainable: bool):
    for blk in (model.enc1, model.enc2, model.enc3):
        for p in blk.parameters():
            p.requires_grad = trainable


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
        tab = batch["tabular"].to(device, non_blocking=True)  # ignored by model
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    seed = int(exp.get("seed", 2026))
    set_seed(seed)

    date_tag = exp.get("date_tag") or datetime.now().strftime("%m_%d_%y")
    run_dir, run_name, ts, git_meta = make_run_dir(
        exp.get("output_dir", "./runs"),
        exp["name"],
        seed,
        date_tag
    )

    # snapshot config + meta
    shutil.copyfile(args.config, run_dir / f"config_{date_tag}.yaml")
    meta = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "command": f"python train_cae_transfer_01_26_2026.py --config {args.config}",
        "git": git_meta,
    }
    with open(run_dir / f"meta_{date_tag}.json", "w") as f:
        json.dump(meta, f, indent=2)

    dcfg = cfg["data"]
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    device = torch.device(tcfg.get("device", "cuda"))
    num_classes = int(mcfg.get("num_classes", 3))
    in_ch = int(mcfg.get("in_ch", 1))

    def make_ds(split_name: str):
        return ADNIDataset(
            csv_path=dcfg["csv_path"],
            split_json=dcfg["split_json"],
            split=split_name,
            image_id_col=dcfg["image_id_col"],
            rid_col=dcfg["rid_col"],
            label_col=dcfg["label_col"],
            image_dir=dcfg["image_dir"],
            image_glob_pattern=dcfg["image_glob_pattern"],
            cont_cols=dcfg.get("cont_cols", ["Age"]),
            cat_cols=dcfg.get("cat_cols", ["Sex"]),
            tabular_standardize=bool(dcfg.get("tabular_standardize", True)),
            sex_encoding=dcfg.get("sex_encoding", "binary"),
            image_scale=dcfg.get("image_scale", None),
        )

    train_ds = make_ds("train")
    val_ds = make_ds("val")
    test_ds = make_ds("test")

    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True,
        num_workers=int(tcfg.get("num_workers", 6)), pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=int(tcfg.get("num_workers", 6)), pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(tcfg["batch_size"]), shuffle=False,
        num_workers=int(tcfg.get("num_workers", 6)), pin_memory=True, drop_last=False
    )

    # build model
    model = CAEClassifier(
        in_ch=in_ch,
        num_classes=num_classes,
        head=str(mcfg.get("head", "linear")),
        head_hidden=int(mcfg.get("head_hidden", 256)),
    )

    copied = load_cae_encoder_weights(model, mcfg["cae_ckpt_path"], in_ch=in_ch)

    freeze_encoder = bool(mcfg.get("freeze_encoder", True))
    set_encoder_trainable(model, trainable=(not freeze_encoder))

    model.to(device)

    # optimizer only trainable params
    lr_head = float(tcfg.get("lr_head", tcfg.get("lr", 3e-4)))
    lr_enc  = float(tcfg.get("lr_encoder", lr_head * 0.1))
    wd      = float(tcfg.get("weight_decay", 0.05))

    enc_params = list(model.enc1.parameters()) + list(model.enc2.parameters()) + list(model.enc3.parameters())
    head_params = list(model.head.parameters())

    params = [
        {"params": [p for p in enc_params if p.requires_grad], "lr": lr_enc},
        {"params": [p for p in head_params if p.requires_grad], "lr": lr_head},
]
    optimizer = torch.optim.AdamW(params, weight_decay=wd)


    use_amp = bool(tcfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None


    metrics_csv = run_dir / f"metrics_epoch_{date_tag}.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "acc", "balanced_acc"])
        w.writeheader()

    best_ba = -1.0
    best_epoch = None

    epochs = int(tcfg.get("epochs", 10))
    for ep in range(1, epochs + 1):
        tr = run_one_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, num_classes=num_classes)
        va = run_one_epoch(model, val_loader, device, optimizer=None, scaler=None, num_classes=num_classes)

        with open(metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "acc", "balanced_acc"])
            w.writerow({"epoch": ep, "split": "train", **tr})
            w.writerow({"epoch": ep, "split": "val", **va})

        print(
            f"[{ep}/{epochs}] "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} ba={tr['balanced_acc']:.4f} | "
            f"val loss={va['loss']:.4f} acc={va['acc']:.4f} ba={va['balanced_acc']:.4f}"
        )

        if va["balanced_acc"] > best_ba:
            best_ba = va["balanced_acc"]
            best_epoch = ep
            torch.save(
                {"model_state": model.state_dict(), "epoch": ep, "best_val_balanced_acc": best_ba},
                run_dir / f"model_best_{date_tag}.pt"
            )

    torch.save({"model_state": model.state_dict(), "epoch": epochs}, run_dir / f"model_final_{date_tag}.pt")

    te = run_one_epoch(model, test_loader, device, optimizer=None, scaler=None, num_classes=num_classes)

    summary = {
        "best_epoch": best_epoch,
        "best_val_balanced_acc": float(best_ba),
        "test": te,
        "freeze_encoder": freeze_encoder,
        "copied_encoder_params": int(copied),
        "cae_ckpt_path": mcfg["cae_ckpt_path"],
    }
    with open(run_dir / f"metrics_{date_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Done]")
    print(" run_dir:", run_dir)
    print(" best_epoch:", best_epoch, "best_val_balanced_acc:", best_ba)
    print(" test:", te)


if __name__ == "__main__":
    main()