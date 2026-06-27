# rigid_flow/infer.py
# -*- coding: utf-8 -*-
"""
Inference & evaluation for rigid fingertip pose.

- Loads config & checkpoint
- Builds validation loader
- Runs rectified-flow residual sampling (RK4), adds to coarse head
- If supervision == 'relative', restores to WORLD frame using ref frame ('last' or 'center')
- Saves:
    runs/<run>/val_preds.csv   (tip_pos_mm + tip_quat[x,y,z,w], WORLD frame)
    runs/<run>/metrics.json    (MAE/RMSE for position, mean/median deg for rotation)
- Also prints a one-line summary at the end.

Run:
  python -m rigid_flow.infer --config configs/rigid_config.yaml --ckpt runs/<run>/model.pt
"""
import os, json, math, yaml, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import RigidSeqDataset
from .models import RigidTipCFM, ModelCfg
from . import geometry as geom  # always module prefix to avoid local shadowing


def auto_device(name: str):
    return "cuda" if (name=="auto" and torch.cuda.is_available()) else ("cpu" if name=="auto" else name)


def geodesic_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """angle (rad) = arccos((trace(Ra^T Rb)-1)/2) for each batch item."""
    M = torch.einsum('bij,bjk->bik', Ra.transpose(1,2), Rb)
    tr = M[:,0,0] + M[:,1,1] + M[:,2,2]
    cos = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cos)


def make_loader(cfg_data, bs, num_workers, mode):
    ds = RigidSeqDataset(
        files=cfg_data["files"],
        schema_file=cfg_data.get("schema_file"),
        window_size=cfg_data["window_size"],
        window_stride=cfg_data["window_stride"],
        sensor_scale=cfg_data.get("sensor_scale", 1024.0),
        stats_path=cfg_data.get("stats_path"),
        mode=mode,
        pos_unit=cfg_data.get("pos_unit", "mm"),
        supervision=cfg_data.get("supervision", "world"),
        ref_frame=cfg_data.get("ref_frame", "last"),
    )
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    return ds, dl


def build_model(cfg):
    m = cfg.get("model", {})
    mcfg = ModelCfg(
        enc_type      = m.get("enc_type", "tfm"),
        seq_hidden    = int(m.get("seq_hidden",128)),
        seq_layers    = int(m.get("seq_layers",2)),
        pooling       = m.get("pooling","mean"),
        tfm_nhead     = int(m.get("tfm_nhead",4)),
        tfm_dropout   = float(m.get("tfm_dropout",0.1)),
        tcn_ksize     = int(m.get("tcn_ksize",3)),
        tcn_dropout   = float(m.get("tcn_dropout",0.1)),
        mamba_d_state = int(m.get("mamba_d_state",16)),
        mamba_d_conv  = int(m.get("mamba_d_conv",4)),
        mamba_expand  = int(m.get("mamba_expand",2)),
        mamba_dropout = float(m.get("mamba_dropout",0.0)),
        head_hidden   = int(m.get("head_hidden",512)),
        head_depth    = int(m.get("head_depth",3)),
        head_act      = m.get("head_act","silu"),
        flow_width    = int(m.get("flow_width",512)),
        flow_depth    = int(m.get("flow_depth",4)),
        flow_tfeat    = int(m.get("flow_tfeat",16)),
        flow_act      = m.get("flow_act","silu"),
    )
    return RigidTipCFM(cfg=mcfg)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt",   type=str, required=True)
    return ap.parse_args()


def main(config_path: str, ckpt_path: str):
    # ---- config & device ----
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = auto_device(cfg.get("device","auto"))

    # ---- data ----
    va_data = {
        "files": cfg["data"]["test_files"],
        "schema_file": cfg["data"].get("schema_file"),
        "window_size": cfg["data"]["window_size"],
        "window_stride": cfg["data"]["window_stride"],
        "sensor_scale": cfg["data"].get("sensor_scale", 1024.0),
        "stats_path": cfg["data"].get("stats_path"),
        "pos_unit": cfg["data"].get("pos_unit","mm"),
        "supervision": cfg["data"].get("supervision","world"),
        "ref_frame": cfg["data"].get("ref_frame","last"),
    }
    _, va_loader = make_loader(va_data, bs=cfg["train"]["batch_size"], num_workers=cfg["train"].get("num_workers",0), mode="test")

    # ---- model & weights ----
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # supports both plain state_dict and EMA shadow stored as "model"
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # ---- infer hyper-params ----
    steps = int(cfg.get("infer",{}).get("steps", 40))
    n_samples = int(cfg.get("infer",{}).get("n_samples", 4))
    dt = 1.0/steps

    supervision = (cfg.get("data",{}).get("supervision","world")).lower()
    ref_frame   = (cfg.get("data",{}).get("ref_frame","last")).lower()

    all_preds = []
    pos_errs = []     # per-sample L2 error in mm (WORLD)
    rot_errs = []     # per-sample geodesic angle in deg (WORLD)

    with torch.no_grad():
        for batch in va_loader:
            back_seq = batch["back_seq"].to(device)     # [B,T,13]
            y9_tgt   = batch["y9_target"].to(device)    # [B,9]
            B = back_seq.size(0)

            # Coarse prediction
            coarse = model.coarse_only(back_seq)        # [B,9]

            # Flow residual sampling (mean of N trajectories)
            def f(x, t):
                v, _ = model(x, t, back_seq)
                return v

            outs = []
            for _ in range(n_samples):
                x = torch.randn(B, 9, device=device)
                t = torch.zeros(B, 1, device=device)
                for _ in range(steps):
                    k1 = f(x, t)
                    k2 = f(x + 0.5*dt*k1, t + 0.5*dt)
                    k3 = f(x + 0.5*dt*k2, t + 0.5*dt)
                    k4 = f(x + dt*k3,   t + dt)
                    x  = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                    t  = t + dt
                outs.append(x)

            r_mean = torch.stack(outs, 0).mean(0)       # [B,9]
            y9p = coarse + r_mean                        # [B,9] predicted (pos3 + r6d), supervision space

            # Predicted rotation
            R_pred = geom.r6d_to_matrix(y9p[:, 3:9])    # [B,3,3]
            pos_pred = y9p[:, :3]                       # [B,3]

            # Target rotation
            R_gt = geom.r6d_to_matrix(y9_tgt[:, 3:9])
            pos_gt = y9_tgt[:, :3]

            # If relative supervision, restore both pred & gt to WORLD
            if supervision == "relative":
                # choose reference index in the window
                if ref_frame == "center":
                    idx = back_seq.shape[1] // 2
                else:
                    idx = back_seq.shape[1] - 1

                p_back = back_seq[:, idx, 0:3]           # [B,3] WORLD (yzx, mm)
                Rb6    = back_seq[:, idx, 3:9]           # [B,6]
                R_back = geom.r6d_to_matrix(Rb6)         # [B,3,3]

                # pred -> world
                pos_world_pred = torch.einsum('bij,bj->bi', R_back, pos_pred) + p_back
                R_world_pred   = torch.einsum('bij,bjk->bik', R_back, R_pred)

                # gt -> world
                pos_world_gt = torch.einsum('bij,bj->bi', R_back, pos_gt) + p_back
                R_world_gt   = torch.einsum('bij,bjk->bik', R_back, R_gt)
            else:
                pos_world_pred = pos_pred
                R_world_pred   = R_pred
                pos_world_gt   = pos_gt
                R_world_gt     = R_gt

            # errors
            pe = torch.linalg.norm(pos_world_pred - pos_world_gt, dim=-1)    # [B], mm
            re = geodesic_angle(R_world_pred, R_world_gt) * (180.0/math.pi) # [B], deg
            pos_errs.append(pe.cpu())
            rot_errs.append(re.cpu())

            # save preds (WORLD frame, mm + quaternion)
            q_pred = geom.matrix_to_quat(R_world_pred)  # [B,4], (x,y,z,w)
            for i in range(B):
                all_preds.append({
                    "tip_px_mm": float(pos_world_pred[i,0]),
                    "tip_py_mm": float(pos_world_pred[i,1]),
                    "tip_pz_mm": float(pos_world_pred[i,2]),
                    "tip_qx": float(q_pred[i,0]),
                    "tip_qy": float(q_pred[i,1]),
                    "tip_qz": float(q_pred[i,2]),
                    "tip_qw": float(q_pred[i,3]),
                })

    # ---- aggregate metrics ----
    pos_errs = torch.cat(pos_errs, 0).double().numpy()
    rot_errs = torch.cat(rot_errs, 0).double().numpy()

    metrics = {
        "count": int(pos_errs.shape[0]),
        "pos_mae_mm": float(np.mean(np.abs(pos_errs))),
        "pos_rmse_mm": float(np.sqrt(np.mean(pos_errs**2))),
        "rot_mean_deg": float(np.mean(rot_errs)),
        "rot_med_deg": float(np.median(rot_errs)),
    }

    # ---- Save outputs ----
    run_dir = os.path.dirname(ckpt_path)
    metrics_path = os.path.join(run_dir, "metrics.json")
    preds_path = os.path.join(run_dir, "val_preds.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved metrics: {metrics_path}")

    # write CSV
    try:
        import pandas as pd
        pd.DataFrame(all_preds).to_csv(preds_path, index=False)
    except Exception:
        # fallback to csv module if pandas is unavailable
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tip_px_mm","tip_py_mm","tip_pz_mm","tip_qx","tip_qy","tip_qz","tip_qw"])
            for r in all_preds:
                w.writerow([r["tip_px_mm"], r["tip_py_mm"], r["tip_pz_mm"],
                            r["tip_qx"], r["tip_qy"], r["tip_qz"], r["tip_qw"]])
    print(f"[OK] Saved predictions: {preds_path}")

    # ---- Pretty print metrics to console ----
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            M = json.load(f)
        print(
            "[Summary] "
            f"N={M.get('count', 'NA')}  "
            f"pos_mae_mm={M.get('pos_mae_mm', float('nan')):.3f}  "
            f"pos_rmse_mm={M.get('pos_rmse_mm', float('nan')):.3f}  "
            f"rot_mean_deg={M.get('rot_mean_deg', float('nan')):.3f}  "
            f"rot_med_deg={M.get('rot_med_deg', float('nan')):.3f}"
        )
    except Exception as e:
        print(f"[WARN] Failed to pretty-print metrics: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.ckpt)
