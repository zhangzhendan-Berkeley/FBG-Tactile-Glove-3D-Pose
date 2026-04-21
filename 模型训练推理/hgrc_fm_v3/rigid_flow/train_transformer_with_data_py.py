# rigid_flow/train_transformer_with_data_py.py
# -*- coding: utf-8 -*-

import os
import json
import math
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import RigidSeqDataset
from rigid_flow import geometry as geom


def auto_device(name: str):
    return "cuda" if (name == "auto" and torch.cuda.is_available()) else ("cpu" if name == "auto" else name)


def geodesic_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """
    Ra, Rb: [B,3,3]
    return: [B] angle in rad
    """
    M = torch.einsum("bij,bjk->bik", Ra.transpose(1, 2), Rb)
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    cos = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cos)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,T,C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SimpleTransformerRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int = 13,
        out_dim: int = 9,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "last",   # "last" | "mean"
    ):
        super().__init__()
        self.pooling = pooling

        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,13]
        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)

        if self.pooling == "mean":
            feat = x.mean(dim=1)
        else:
            feat = x[:, -1, :]

        y = self.head(feat)  # [B,9]
        return y


def make_loader(cfg_data, bs, shuffle, num_workers, mode):
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
        augment=cfg_data.get("augment", None),
    )
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    return ds, dl


class SeqStandardizer:
    """
    对 back_seq [T,13] 和 y9_target [9] 做标准化
    """
    def __init__(self):
        self.x_mean = None   # [1,1,13]
        self.x_std = None
        self.y_mean = None   # [1,9]
        self.y_std = None

    def fit(self, ds: RigidSeqDataset):
        x_list = []
        y_list = []
        for i in range(len(ds)):
            item = ds[i]
            x_list.append(item["back_seq"])     # [T,13]
            y_list.append(item["y9_target"])    # [9]

        X = torch.stack(x_list, dim=0)          # [N,T,13]
        Y = torch.stack(y_list, dim=0)          # [N,9]

        self.x_mean = X.mean(dim=(0, 1), keepdim=True)   # [1,1,13]
        self.x_std = X.std(dim=(0, 1), keepdim=True)     # [1,1,13]
        self.x_std = torch.clamp(self.x_std, min=1e-6)

        self.y_mean = Y.mean(dim=0, keepdim=True)        # [1,9]
        self.y_std = Y.std(dim=0, keepdim=True)          # [1,9]
        self.y_std = torch.clamp(self.y_std, min=1e-6)

    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

    def inverse_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


def pose_loss_std(pred_std: torch.Tensor, tgt_std: torch.Tensor):
    """
    标准化空间里的损失：
    - 位置：L1
    - 旋转：rot6d 的 L1
    """
    pred_pos = pred_std[:, :3]
    tgt_pos = tgt_std[:, :3]

    pred_rot6 = pred_std[:, 3:9]
    tgt_rot6 = tgt_std[:, 3:9]

    loss_pos = F.l1_loss(pred_pos, tgt_pos)
    loss_rot = F.l1_loss(pred_rot6, tgt_rot6)
    loss = loss_pos + loss_rot

    return loss, {
        "loss_pos": float(loss_pos.item()),
        "loss_rot6": float(loss_rot.item()),
    }


@torch.no_grad()
def evaluate(model, loader, device, supervision: str, ref_frame: str, scaler: SeqStandardizer):
    model.eval()

    total_loss = 0.0
    total_n = 0

    pos_l2_list = []
    pos_xyz_mae_list = []
    rot_deg_list = []

    for batch in loader:
        back_seq = batch["back_seq"].to(device)     # [B,T,13]
        y9_tgt = batch["y9_target"].to(device)      # [B,9]
        B = back_seq.size(0)

        x_std = scaler.transform_x(back_seq)
        y_std = scaler.transform_y(y9_tgt)

        pred_std = model(x_std)
        loss, _ = pose_loss_std(pred_std, y_std)

        pred = scaler.inverse_y(pred_std)

        pred_pos = pred[:, :3]
        pred_R = geom.r6d_to_matrix(pred[:, 3:9])

        tgt_pos = y9_tgt[:, :3]
        tgt_R = geom.r6d_to_matrix(y9_tgt[:, 3:9])

        if supervision.lower() == "relative":
            idx = (back_seq.shape[1] // 2) if (ref_frame.lower() == "center") else (back_seq.shape[1] - 1)

            p_back = back_seq[:, idx, 0:3]
            R_back = geom.r6d_to_matrix(back_seq[:, idx, 3:9])

            pred_pos = torch.einsum("bij,bj->bi", R_back, pred_pos) + p_back
            pred_R = torch.einsum("bij,bjk->bik", R_back, pred_R)

            tgt_pos = torch.einsum("bij,bj->bi", R_back, tgt_pos) + p_back
            tgt_R = torch.einsum("bij,bjk->bik", R_back, tgt_R)

        pos_l2 = torch.linalg.norm(pred_pos - tgt_pos, dim=-1)              # [B]
        pos_xyz_mae = torch.abs(pred_pos - tgt_pos).mean(dim=-1)            # [B]
        rot_deg = geodesic_angle(pred_R, tgt_R) * (180.0 / math.pi)

        pos_l2_list.append(pos_l2.cpu())
        pos_xyz_mae_list.append(pos_xyz_mae.cpu())
        rot_deg_list.append(rot_deg.cpu())

        total_loss += float(loss.item()) * B
        total_n += B

    pos_l2_all = torch.cat(pos_l2_list, dim=0).numpy()
    pos_xyz_mae_all = torch.cat(pos_xyz_mae_list, dim=0).numpy()
    rot_deg_all = torch.cat(rot_deg_list, dim=0).numpy()

    return {
        "loss": total_loss / max(total_n, 1),
        "pos_mae_mm": float(np.mean(pos_xyz_mae_all)),     # 真正的xyz平均绝对误差
        "pos_l2_mm": float(np.mean(pos_l2_all)),           # 平均欧氏距离
        "pos_rmse_mm": float(np.sqrt(np.mean(pos_l2_all ** 2))),
        "rot_mean_deg": float(np.mean(rot_deg_all)),
        "rot_med_deg": float(np.median(rot_deg_all)),
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = auto_device(cfg.get("device", "auto"))

    run_dir = os.path.join("runs", "tf_from_data_py_std")
    os.makedirs(run_dir, exist_ok=True)

    tr_data = {
        "files": cfg["data"]["train_files"],
        "schema_file": cfg["data"].get("schema_file"),
        "window_size": cfg["data"]["window_size"],
        "window_stride": cfg["data"]["window_stride"],
        "sensor_scale": cfg["data"].get("sensor_scale", 1024.0),
        "stats_path": cfg["data"].get("stats_path"),
        "pos_unit": cfg["data"].get("pos_unit", "mm"),
        "supervision": cfg["data"].get("supervision", "world"),
        "ref_frame": cfg["data"].get("ref_frame", "last"),
        "augment": None,
    }
    va_data = {
        "files": cfg["data"]["test_files"],
        "schema_file": cfg["data"].get("schema_file"),
        "window_size": cfg["data"]["window_size"],
        "window_stride": cfg["data"]["window_stride"],
        "sensor_scale": cfg["data"].get("sensor_scale", 1024.0),
        "stats_path": cfg["data"].get("stats_path"),
        "pos_unit": cfg["data"].get("pos_unit", "mm"),
        "supervision": cfg["data"].get("supervision", "world"),
        "ref_frame": cfg["data"].get("ref_frame", "last"),
    }

    tr_ds, tr_loader = make_loader(
        tr_data,
        bs=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        mode="train",
    )
    va_ds, va_loader = make_loader(
        va_data,
        bs=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        mode="test",
    )

    print("len(tr_ds) =", len(tr_ds))
    print("len(va_ds) =", len(va_ds))
    one = tr_ds[0]
    print("back_seq shape:", one["back_seq"].shape)
    print("y9_target shape:", one["y9_target"].shape)
    print("last frame sensors:", one["back_seq"][-1, -4:])

    # ===== fit scaler on train set =====
    scaler = SeqStandardizer()
    print("Fitting scaler on training dataset...")
    scaler.fit(tr_ds)
    print("x_mean last 4 (sensors):", scaler.x_mean[0, 0, -4:])
    print("x_std  last 4 (sensors):", scaler.x_std[0, 0, -4:])
    print("y_mean[:3] =", scaler.y_mean[0, :3])
    print("y_std[:3]  =", scaler.y_std[0, :3])

    model = SimpleTransformerRegressor(
        in_dim=13,
        out_dim=9,
        d_model=cfg.get("tf_test", {}).get("d_model", 128),
        nhead=cfg.get("tf_test", {}).get("nhead", 4),
        num_layers=cfg.get("tf_test", {}).get("num_layers", 3),
        ff_dim=cfg.get("tf_test", {}).get("ff_dim", 256),
        dropout=cfg.get("tf_test", {}).get("dropout", 0.1),
        pooling=cfg.get("tf_test", {}).get("pooling", "last"),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    epochs = int(cfg["train"].get("epochs", 20))
    best_val = 1e18
    best_path = os.path.join(run_dir, "best_model.pt")

    supervision = cfg["data"].get("supervision", "world")
    ref_frame = cfg["data"].get("ref_frame", "last")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        steps = 0

        for batch in tr_loader:
            back_seq = batch["back_seq"].to(device)
            y9_tgt = batch["y9_target"].to(device)

            x_std = scaler.transform_x(back_seq)
            y_std = scaler.transform_y(y9_tgt)

            pred_std = model(x_std)
            loss, info = pose_loss_std(pred_std, y_std)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            steps += 1

        tr_loss = total / max(steps, 1)
        val_metrics = evaluate(model, va_loader, device, supervision=supervision, ref_frame=ref_frame, scaler=scaler)

        print(
            f"[Epoch {ep:03d}] "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_pos_mae_mm={val_metrics['pos_mae_mm']:.3f} | "
            f"val_pos_l2_mm={val_metrics['pos_l2_mm']:.3f} | "
            f"val_rot_mean_deg={val_metrics['rot_mean_deg']:.3f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "x_mean": scaler.x_mean.cpu(),
                    "x_std": scaler.x_std.cpu(),
                    "y_mean": scaler.y_mean.cpu(),
                    "y_std": scaler.y_std.cpu(),
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    scaler.x_mean = ckpt["x_mean"]
    scaler.x_std = ckpt["x_std"]
    scaler.y_mean = ckpt["y_mean"]
    scaler.y_std = ckpt["y_std"]

    test_metrics = evaluate(model, va_loader, device, supervision=supervision, ref_frame=ref_frame, scaler=scaler)

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print("\n===== Final Result =====")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)