# rigid_flow/train_recent_baselines.py
# -*- coding: utf-8 -*-
"""Train lightweight recent time-series baselines for fingertip pose regression."""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import RigidSeqDataset
from .train_mamba_coarse_only import SeqStandardizer, evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device(name: str):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def make_dataset(cfg, split: str, mode: str):
    data_cfg = cfg["data"]
    ds = RigidSeqDataset(
        files=data_cfg[f"{split}_files"],
        schema_file=data_cfg.get("schema_file"),
        window_size=data_cfg["window_size"],
        window_stride=data_cfg["window_stride"],
        sensor_scale=data_cfg.get("sensor_scale", 1.0),
        stats_path=data_cfg.get("stats_path"),
        mode=mode,
        pos_unit=data_cfg.get("pos_unit", "mm"),
        supervision=data_cfg.get("supervision", "world"),
        ref_frame=data_cfg.get("ref_frame", "last"),
        augment=None,
    )
    return ds


class PoseHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 9),
        )

    def forward(self, x):
        return self.net(x)


class PatchTSTPoseRegressor(nn.Module):
    """Patch-token Transformer adapted from PatchTST for sequence-to-pose regression."""

    def __init__(
        self,
        in_dim=13,
        seq_len=96,
        patch_len=16,
        patch_stride=8,
        d_model=128,
        nhead=4,
        num_layers=3,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        n_patches = 1 + (seq_len - patch_len) // patch_stride
        self.patch_proj = nn.Linear(in_dim * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = PoseHead(d_model, hidden=256, dropout=dropout)

    def forward(self, x):
        # [B,T,13] -> [B,Npatch,patch_len,13] -> [B,Npatch,d_model]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).contiguous().flatten(start_dim=2)
        h = self.patch_proj(patches)
        h = self.encoder(h + self.pos_embed[:, : h.size(1)])
        return self.head(h.mean(dim=1))

    def coarse_only(self, x):
        return self.forward(x)


class ModernTCNBlock(nn.Module):
    """Lightweight large-kernel depthwise-convolution block inspired by ModernTCN."""

    def __init__(self, d_model=128, kernel_size=15, expansion=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm(x)
        h = self.dwconv(h.transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(h)


class ModernTCNPoseRegressor(nn.Module):
    """Modern large-kernel temporal convolution encoder for pose regression."""

    def __init__(self, in_dim=13, d_model=128, num_layers=4, kernel_size=15, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList(
            [ModernTCNBlock(d_model, kernel_size, expansion=2, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = PoseHead(d_model, hidden=256, dropout=dropout)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.out_norm(h).mean(dim=1))

    def coarse_only(self, x):
        return self.forward(x)


class BiLSTMPoseRegressor(nn.Module):
    """Large bidirectional LSTM baseline for sequence-to-pose regression."""

    def __init__(
        self,
        in_dim=13,
        hidden_size=512,
        num_layers=4,
        dropout=0.1,
        head_hidden=512,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(hidden_size * 2)
        self.head = PoseHead(hidden_size * 2, hidden=head_hidden, dropout=dropout)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(self.out_norm(h.mean(dim=1)))

    def coarse_only(self, x):
        return self.forward(x)


def pose_loss_std(pred_std, target_std):
    return F.l1_loss(pred_std[:, :3], target_std[:, :3]) + F.l1_loss(
        pred_std[:, 3:9], target_std[:, 3:9]
    )


def build_model(name: str, seq_len: int):
    if name == "patchtst":
        return PatchTSTPoseRegressor(seq_len=seq_len)
    if name == "moderntcn":
        return ModernTCNPoseRegressor()
    if name == "bilstm_large":
        return BiLSTMPoseRegressor()
    raise ValueError(f"Unknown baseline: {name}")


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 0)))
    device = auto_device(cfg.get("device", "auto"))

    train_ds = make_dataset(cfg, "train", "train")
    val_ds = make_dataset(cfg, "val", "test")
    test_ds = make_dataset(cfg, "test", "test")
    batch_size = int(cfg["train"]["batch_size"])
    workers = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    scaler = SeqStandardizer()
    scaler.fit(train_ds)

    model = build_model(args.model, int(cfg["data"]["window_size"])).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 5e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = args.epochs or int(cfg["train"].get("epochs", 20))
    supervision = cfg["data"].get("supervision", "world")
    ref_frame = cfg["data"].get("ref_frame", "last")

    run_dir = os.path.join("runs", args.model)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_model.pt")
    best_val = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            x = scaler.transform_x(batch["back_seq"].to(device))
            y = scaler.transform_y(batch["y9_target"].to(device))
            loss = pose_loss_std(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        val_metrics = evaluate(model, val_loader, device, supervision, ref_frame, scaler)
        print(
            f"[{args.model} {epoch:03d}/{epochs}] train={total_loss/max(steps,1):.4f} "
            f"val_l2={val_metrics['pos_l2_mm']:.3f} val_rot={val_metrics['rot_mean_deg']:.3f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "x_mean": scaler.x_mean,
                    "x_std": scaler.x_std,
                    "y_mean": scaler.y_mean,
                    "y_std": scaler.y_std,
                    "parameter_count": parameter_count,
                    "cfg": cfg,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    scaler.x_mean, scaler.x_std = ckpt["x_mean"], ckpt["x_std"]
    scaler.y_mean, scaler.y_std = ckpt["y_mean"], ckpt["y_std"]
    metrics = evaluate(model, test_loader, device, supervision, ref_frame, scaler)
    metrics["parameter_count"] = parameter_count
    metrics["selection_split"] = "validation"
    metrics["evaluation_split"] = "test"
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rigid_config.yaml")
    parser.add_argument("--model", required=True, choices=["patchtst", "moderntcn", "bilstm_large"])
    parser.add_argument("--epochs", type=int, default=None)
    main(parser.parse_args())
