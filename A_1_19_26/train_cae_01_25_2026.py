# A_1_19_26/train_cae_01_25_2026.py
# 01-25-2026 Roy Seo & ChatGPT
#
# Purpose:
#   - Unsupervised CAE pretraining (reconstruction) on skull-stripped 3D T1 MRI.
#   - Uses ADNIDataset to load images; ignores label/tabular.
#
# Input:
#   - image: (B, 1, 64, 64, 64)  (your current data)
#
# Output (run_dir):
#   - model_best_<date>.pt  (best by val_loss)
#   - model_final_<date>.pt
#   - metrics_<date>.json
#   - metrics_epoch_<date>.csv
#
# Run:
#   cd ~/royseo/projects/adni-mamba/A_1_19_26
#   conda activate hope-adni-mamba-gpu
#   python train_cae_01_25_2026.py --config ./configs/pretrain/cae_pretrain_full_01_25_2026.yaml

import argparse, csv, json, os, shutil, subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.adni_dataset_1_19_26 import ADNIDataset
from src.models_rs.cae_3d_01_24_2026 import CAE3D, masked_mse


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    git_meta = get_git_metadata()
    gh = git_meta.get("git_hash_short") or "nogit"

    base = Path(output_dir) / exp_name
    base.mkdir(parents=True, exist_ok=True)

    run_name = f"{exp_name}_{date_tag}_{ts}_seed{seed}_{gh}"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name, ts, git_meta


@torch.no_grad()
def eval_one_epoch(model, loader, device, use_mask: bool = False):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True).float()

        # per-volume z-score (helps stability a lot in reconstruction)
        x = (x - x.mean(dim=(2,3,4), keepdim=True)) / (x.std(dim=(2,3,4), keepdim=True) + 1e-6)

        x_hat = model(x)
        loss = masked_mse(x_hat, x, mask=None) if not use_mask else masked_mse(x_hat, x, mask=batch.get("mask", None))
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def train_one_epoch(model, loader, device, optimizer, scaler, use_mask: bool = False):
    model.train(True)
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True).float()

        # per-volume z-score
        x = (x - x.mean(dim=(2,3,4), keepdim=True)) / (x.std(dim=(2,3,4), keepdim=True) + 1e-6)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            x_hat = model(x)
            loss = masked_mse(x_hat, x, mask=None) if not use_mask else masked_mse(x_hat, x, mask=batch.get("mask", None))

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    exp, dcfg, tcfg, mcfg = cfg["experiment"], cfg["data"], cfg["train"], cfg["model"]

    exp_name = exp["name"]
    seed = int(exp.get("seed", 0))
    date_tag = datetime.now().strftime("%m_%d_%y")
    set_seed(seed)

    run_dir, run_name, ts, git_meta = make_run_dir(exp["output_dir"], exp_name, seed, date_tag)

    # snapshot config + split
    shutil.copy2(args.config, run_dir / f"config_{date_tag}.yaml")
    if os.path.exists(dcfg["split_json"]):
        shutil.copy2(dcfg["split_json"], run_dir / f"split_{date_tag}.json")

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
    with open(run_dir / f"meta_{date_tag}.json", "w") as f:
        json.dump(meta, f, indent=2)

    device_req = str(tcfg.get("device", "cuda"))
    device = torch.device(device_req if (device_req.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # dataset (reuse ADNIDataset; ignore labels/tabular in training loop)
    def make_ds(split_name: str):
        return ADNIDataset(
            csv_path=dcfg["csv_path"],
            split_json=dcfg["split_json"],
            split=split_name,
            image_id_col=dcfg["image_id_col"],
            image_dir=dcfg["image_dir"],
            image_glob_pattern=dcfg["image_glob_pattern"],
            rid_col=dcfg["rid_col"],
            label_col=dcfg["label_col"],      # can be DX2; not used for loss
            cont_cols=dcfg.get("cont_cols", None),
            cat_cols=dcfg.get("cat_cols", None),
            tabular_standardize=bool(dcfg.get("tabular_standardize", True)),
            sex_encoding=dcfg.get("sex_encoding", "binary"),
        )

    train_ds = make_ds("train")
    val_ds   = make_ds("val")

    bs = int(tcfg["batch_size"])
    nw = int(tcfg.get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    model = CAE3D(in_ch=int(mcfg.get("in_ch", 1))).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(tcfg.get("amp", True)) and device.type == "cuda")

    epochs = int(tcfg["epochs"])
    best_val = float("inf")
    best_epoch = -1

    ckpt_best = run_dir / f"cae_best_{date_tag}.pt"
    ckpt_final = run_dir / f"cae_final_{date_tag}.pt"
    metrics_json = run_dir / f"metrics_{date_tag}.json"
    metrics_csv = run_dir / f"metrics_epoch_{date_tag}.csv"

    rows = []
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, opt, scaler, use_mask=False)
        va_loss = eval_one_epoch(model, val_loader, device, use_mask=False)

        rows.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"[{ep}/{epochs}] train_mse={tr_loss:.6f} | val_mse={va_loss:.6f}")

        # save best by val loss
        if va_loss < best_val:
            best_val = float(va_loss)
            best_epoch = ep
            torch.save(
                {
                    "epoch": ep,
                    "val_loss": best_val,
                    "model_state": model.state_dict(),
                    # convenience: store encoder-only state too
                    "encoder_state": {
                        "enc1": model.enc1.state_dict(),
                        "enc2": model.enc2.state_dict(),
                        "enc3": model.enc3.state_dict(),
                    },
                },
                ckpt_best,
            )

    torch.save({"epoch": epochs, "model_state": model.state_dict()}, ckpt_final)

    # write metrics files
    with open(metrics_json, "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_mse": best_val, "epochs": epochs, "batch_size": bs}, f, indent=2)

    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n[Saved run]")
    print(" run_dir:", run_dir)
    print(" best_ckpt:", ckpt_best)
    print(" final_ckpt:", ckpt_final)
    print(" best_epoch:", best_epoch, "best_val_mse:", best_val)


if __name__ == "__main__":
    main()