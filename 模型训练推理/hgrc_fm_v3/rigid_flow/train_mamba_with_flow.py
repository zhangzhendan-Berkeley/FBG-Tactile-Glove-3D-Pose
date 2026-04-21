# rigid_flow/train_mamba_with_flow.py
# -*- coding: utf-8 -*-

import os
import json
import math
import yaml
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import RigidSeqDataset
from .models import RigidTipCFM, ModelCfg
from rigid_flow import geometry as geom


def auto_device(name: str):
    return "cuda" if (name == "auto" and torch.cuda.is_available()) else ("cpu" if name == "auto" else name)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def geodesic_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """
    Ra, Rb: [B,3,3]
    return: [B] angle in rad
    """
    M = torch.einsum("bij,bjk->bik", Ra.transpose(1, 2), Rb)
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    cos = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cos)


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
        augment=None,   # flow 阶段先不做增强
    )
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    return ds, dl


class SeqStandardizer:
    def __init__(self):
        self.x_mean = None   # [1,1,13]
        self.x_std = None
        self.y_mean = None   # [1,9]
        self.y_std = None

    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

    def inverse_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)


def build_model_from_cfg(cfg) -> RigidTipCFM:
    m = cfg.get("model", {})
    mcfg = ModelCfg(
        enc_type="mamba",
        seq_hidden=int(m.get("seq_hidden", 256)),
        seq_layers=int(m.get("seq_layers", 4)),
        pooling=m.get("pooling", "mean"),
        tfm_nhead=int(m.get("tfm_nhead", 4)),
        tfm_dropout=float(m.get("tfm_dropout", 0.1)),
        tcn_ksize=int(m.get("tcn_ksize", 3)),
        tcn_dropout=float(m.get("tcn_dropout", 0.1)),
        mamba_d_state=int(m.get("mamba_d_state", 16)),
        mamba_d_conv=int(m.get("mamba_d_conv", 4)),
        mamba_expand=int(m.get("mamba_expand", 2)),
        mamba_dropout=float(m.get("mamba_dropout", 0.0)),
        head_hidden=int(m.get("head_hidden", 512)),
        head_depth=int(m.get("head_depth", 3)),
        head_act=m.get("head_act", "silu"),
        flow_width=int(m.get("flow_width", 512)),
        flow_depth=int(m.get("flow_depth", 4)),
        flow_tfeat=int(m.get("flow_tfeat", 16)),
        flow_act=m.get("flow_act", "silu"),
    )
    return RigidTipCFM(cfg=mcfg)


def freeze_coarse_branch(model: RigidTipCFM):
    """
    冻结 coarse 分支：encoder + head
    flow 分支继续训练
    """
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = False


def count_trainable_params(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


def compute_pose_metrics_from_std(
    pred_std: torch.Tensor,
    y9_tgt: torch.Tensor,
    back_seq: torch.Tensor,
    scaler: SeqStandardizer,
    supervision: str,
    ref_frame: str,
):
    """
    pred_std: [B,9] in standardized space
    y9_tgt : [B,9] raw space
    back_seq: [B,T,13] raw space
    """
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

    pos_l2 = torch.linalg.norm(pred_pos - tgt_pos, dim=-1)
    pos_xyz_mae = torch.abs(pred_pos - tgt_pos).mean(dim=-1)
    rot_deg = geodesic_angle(pred_R, tgt_R) * (180.0 / math.pi)

    return pos_xyz_mae, pos_l2, rot_deg


@torch.no_grad()
def sample_flow_residual(model, x_std, steps: int, n_samples: int):
    """
    用 RK4 对 residual 做 ODE 采样
    返回: [B,9] residual in standardized space
    """
    B = x_std.size(0)
    dt = 1.0 / steps

    def f(x, t):
        v, _ = model(x, t, x_std)
        return v

    outs = []
    for _ in range(n_samples):
        # x = torch.randn(B, 9, device=x_std.device)
        x = torch.zeros(B, 9, device=x_std.device)
        tt = torch.zeros(B, 1, device=x_std.device)

        for _ in range(steps):
            k1 = f(x, tt)
            k2 = f(x + 0.5 * dt * k1, tt + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, tt + 0.5 * dt)
            k4 = f(x + dt * k3, tt + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tt = tt + dt

        outs.append(x)

    return torch.stack(outs, dim=0).mean(dim=0)   # [B,9]


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    supervision: str,
    ref_frame: str,
    scaler: SeqStandardizer,
    steps: int,
    n_samples: int,
    alpha: float = 1.0,
):
    model.eval()

    coarse_pos_mae_list = []
    coarse_pos_l2_list = []
    coarse_rot_deg_list = []

    full_pos_mae_list = []
    full_pos_l2_list = []
    full_rot_deg_list = []

    flow_mse_list = []

    for batch in loader:
        back_seq = batch["back_seq"].to(device)   # raw
        y9_tgt = batch["y9_target"].to(device)    # raw

        x_std = scaler.transform_x(back_seq)
        y_std = scaler.transform_y(y9_tgt)

        # coarse-only
        coarse_std = model.coarse_only(x_std)

        c_pos_mae, c_pos_l2, c_rot_deg = compute_pose_metrics_from_std(
            coarse_std, y9_tgt, back_seq, scaler, supervision, ref_frame
        )

        coarse_pos_mae_list.append(c_pos_mae.cpu())
        coarse_pos_l2_list.append(c_pos_l2.cpu())
        coarse_rot_deg_list.append(c_rot_deg.cpu())

        # flow residual target
        res_target = y_std - coarse_std

        # 采样得到 residual

        res_pred = sample_flow_residual(model, x_std, steps=steps, n_samples=n_samples)
        pred_std = coarse_std + alpha * res_pred

        f_pos_mae, f_pos_l2, f_rot_deg = compute_pose_metrics_from_std(
            pred_std, y9_tgt, back_seq, scaler, supervision, ref_frame
        )

        full_pos_mae_list.append(f_pos_mae.cpu())
        full_pos_l2_list.append(f_pos_l2.cpu())
        full_rot_deg_list.append(f_rot_deg.cpu())

        flow_mse = ((res_pred - res_target) ** 2).mean(dim=-1)
        flow_mse_list.append(flow_mse.cpu())

    def agg(xlist):
        x = torch.cat(xlist, dim=0).numpy()
        return float(np.mean(x)), float(np.median(x))

    coarse_pos_mae, _ = agg(coarse_pos_mae_list)
    coarse_pos_l2, _ = agg(coarse_pos_l2_list)
    coarse_rot_mean, coarse_rot_med = agg(coarse_rot_deg_list)

    full_pos_mae, _ = agg(full_pos_mae_list)
    full_pos_l2, _ = agg(full_pos_l2_list)
    full_rot_mean, full_rot_med = agg(full_rot_deg_list)

    flow_mse_mean, _ = agg(flow_mse_list)

    return {
        "coarse": {
            "pos_mae_mm": coarse_pos_mae,
            "pos_l2_mm": coarse_pos_l2,
            "rot_mean_deg": coarse_rot_mean,
            "rot_med_deg": coarse_rot_med,
        },
        "coarse_plus_flow": {
            "pos_mae_mm": full_pos_mae,
            "pos_l2_mm": full_pos_l2,
            "rot_mean_deg": full_rot_mean,
            "rot_med_deg": full_rot_med,
        },
        "flow": {
            "residual_mse": flow_mse_mean,
        }
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--coarse_ckpt", type=str, default=None)
    return ap.parse_args()


def main(config_path: str, coarse_ckpt_path: str | None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 0)))
    device = auto_device(cfg.get("device", "auto"))

    run_dir = os.path.join("runs", "mamba_with_flow")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

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

    # coarse checkpoint 路径
    if coarse_ckpt_path is None:
        coarse_ckpt_path = cfg.get("train", {}).get("coarse_ckpt", None)
    if coarse_ckpt_path is None:
        raise ValueError("Please provide --coarse_ckpt or set train.coarse_ckpt in yaml.")

    print("Loading coarse checkpoint from:", coarse_ckpt_path)

    model = build_model_from_cfg(cfg).to(device)
    coarse_ckpt = torch.load(coarse_ckpt_path, map_location=device)

    msg = model.load_state_dict(coarse_ckpt["model"], strict=False)
    print("missing_keys:", msg.missing_keys)
    print("unexpected_keys:", msg.unexpected_keys)

    # scaler 从 coarse ckpt 恢复
    scaler = SeqStandardizer()
    scaler.x_mean = coarse_ckpt["x_mean"]
    scaler.x_std = coarse_ckpt["x_std"]
    scaler.y_mean = coarse_ckpt["y_mean"]
    scaler.y_std = coarse_ckpt["y_std"]

    print("Loaded scaler from coarse checkpoint.")
    print("x_mean last 4 (sensors):", scaler.x_mean[0, 0, -4:])
    print("x_std  last 4 (sensors):", scaler.x_std[0, 0, -4:])
    print("y_mean[:3] =", scaler.y_mean[0, :3])
    print("y_std[:3]  =", scaler.y_std[0, :3])

    for i in range(13):
        writer.add_scalar(f"scaler/x_mean_{i}", float(scaler.x_mean[0, 0, i].item()), 0)
        writer.add_scalar(f"scaler/x_std_{i}", float(scaler.x_std[0, 0, i].item()), 0)
    for i in range(9):
        writer.add_scalar(f"scaler/y_mean_{i}", float(scaler.y_mean[0, i].item()), 0)
        writer.add_scalar(f"scaler/y_std_{i}", float(scaler.y_std[0, i].item()), 0)

    # 冻结 coarse 分支
    freeze_coarse_branch(model)
    trainable_count = count_trainable_params(model)
    print("Trainable params after freezing coarse branch:", trainable_count)
    writer.add_scalar("model/trainable_params", float(trainable_count), 0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters left after freezing coarse branch.")

    opt = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["train"].get("lr", 1e-4)),  # flow 阶段建议小一点
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    epochs = int(cfg["train"].get("epochs", 20))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))
    lambda_flow = float(cfg.get("flow_train", {}).get("lambda_flow", 1.0))

    infer_steps = int(cfg.get("infer", {}).get("steps", 20))
    infer_n_samples = int(cfg.get("infer", {}).get("n_samples", 1))
    infer_alpha = float(cfg.get("infer", {}).get("alpha", 0.2))

    best_val = 1e18
    best_path = os.path.join(run_dir, "best_model.pt")

    supervision = cfg["data"].get("supervision", "world")
    ref_frame = cfg["data"].get("ref_frame", "last")

    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_flow = 0.0
        steps_count = 0

        for batch in tr_loader:
            back_seq = batch["back_seq"].to(device)   # raw
            y9_tgt = batch["y9_target"].to(device)    # raw
            B = back_seq.size(0)

            x_std = scaler.transform_x(back_seq)
            y_std = scaler.transform_y(y9_tgt)

            # 冻结的 coarse 作为常量 target 基础
            with torch.no_grad():
                coarse_std = model.coarse_only(x_std)

            res_target = y_std - coarse_std

            # rectified flow training target
            x0 = torch.randn_like(res_target)
            t = torch.rand(B, 1, device=device)

            xt = (1.0 - t) * x0 + t * res_target
            u = res_target - x0

            v, _ = model(xt, t, x_std)
            loss_flow = ((v - u) ** 2).mean()

            loss = lambda_flow * loss_flow

            opt.zero_grad()
            loss.backward()

            grad_norm = 0.0
            if grad_clip > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip))
            else:
                # 不裁剪时也顺手统计一下 grad norm
                total_norm_sq = 0.0
                for p in trainable_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm_sq += param_norm.item() ** 2
                grad_norm = total_norm_sq ** 0.5

            opt.step()

            total_loss += float(loss.item())
            total_flow += float(loss_flow.item())
            steps_count += 1
            global_step += 1

            writer.add_scalar("step/train_loss", float(loss.item()), global_step)
            writer.add_scalar("step/flow_loss", float(loss_flow.item()), global_step)
            writer.add_scalar("step/lr", float(opt.param_groups[0]["lr"]), global_step)
            writer.add_scalar("step/grad_norm", float(grad_norm), global_step)

            # 看 residual target 自身尺度，判断 flow 学习难度
            writer.add_scalar("step/res_target_abs_mean", float(res_target.abs().mean().item()), global_step)
            writer.add_scalar("step/res_target_l2_mean", float(torch.linalg.norm(res_target, dim=-1).mean().item()), global_step)

        train_loss = total_loss / max(steps_count, 1)
        train_flow = total_flow / max(steps_count, 1)

        val_metrics = evaluate(
            model=model,
            loader=va_loader,
            device=device,
            supervision=supervision,
            ref_frame=ref_frame,
            scaler=scaler,
            steps=infer_steps,
            n_samples=infer_n_samples,
            alpha=infer_alpha,
        )

        print(
            f"[Epoch {ep:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"flow_loss={train_flow:.6f} | "
            f"coarse_pos_mae={val_metrics['coarse']['pos_mae_mm']:.3f} | "
            f"full_pos_mae={val_metrics['coarse_plus_flow']['pos_mae_mm']:.3f} | "
            f"coarse_rot={val_metrics['coarse']['rot_mean_deg']:.3f} | "
            f"full_rot={val_metrics['coarse_plus_flow']['rot_mean_deg']:.3f}"
        )

        writer.add_scalar("epoch/train_loss", train_loss, ep)
        writer.add_scalar("epoch/train_flow_loss", train_flow, ep)

        writer.add_scalar("epoch/coarse_pos_mae_mm", val_metrics["coarse"]["pos_mae_mm"], ep)
        writer.add_scalar("epoch/coarse_pos_l2_mm", val_metrics["coarse"]["pos_l2_mm"], ep)
        writer.add_scalar("epoch/coarse_rot_mean_deg", val_metrics["coarse"]["rot_mean_deg"], ep)
        writer.add_scalar("epoch/coarse_rot_med_deg", val_metrics["coarse"]["rot_med_deg"], ep)

        writer.add_scalar("epoch/full_pos_mae_mm", val_metrics["coarse_plus_flow"]["pos_mae_mm"], ep)
        writer.add_scalar("epoch/full_pos_l2_mm", val_metrics["coarse_plus_flow"]["pos_l2_mm"], ep)
        writer.add_scalar("epoch/full_rot_mean_deg", val_metrics["coarse_plus_flow"]["rot_mean_deg"], ep)
        writer.add_scalar("epoch/full_rot_med_deg", val_metrics["coarse_plus_flow"]["rot_med_deg"], ep)

        writer.add_scalar("epoch/flow_residual_mse", val_metrics["flow"]["residual_mse"], ep)

        # 直接看 flow 带来的提升
        pos_mae_gain = val_metrics["coarse"]["pos_mae_mm"] - val_metrics["coarse_plus_flow"]["pos_mae_mm"]
        pos_l2_gain = val_metrics["coarse"]["pos_l2_mm"] - val_metrics["coarse_plus_flow"]["pos_l2_mm"]
        rot_gain = val_metrics["coarse"]["rot_mean_deg"] - val_metrics["coarse_plus_flow"]["rot_mean_deg"]

        writer.add_scalar("improve/pos_mae_mm_gain", pos_mae_gain, ep)
        writer.add_scalar("improve/pos_l2_mm_gain", pos_l2_gain, ep)
        writer.add_scalar("improve/rot_mean_deg_gain", rot_gain, ep)

        writer.add_scalars(
            "compare/pos_mae_mm",
            {
                "coarse": val_metrics["coarse"]["pos_mae_mm"],
                "coarse_plus_flow": val_metrics["coarse_plus_flow"]["pos_mae_mm"],
            },
            ep,
        )
        writer.add_scalars(
            "compare/pos_l2_mm",
            {
                "coarse": val_metrics["coarse"]["pos_l2_mm"],
                "coarse_plus_flow": val_metrics["coarse_plus_flow"]["pos_l2_mm"],
            },
            ep,
        )
        writer.add_scalars(
            "compare/rot_mean_deg",
            {
                "coarse": val_metrics["coarse"]["rot_mean_deg"],
                "coarse_plus_flow": val_metrics["coarse_plus_flow"]["rot_mean_deg"],
            },
            ep,
        )

        # 以 coarse+flow 为准选 best
        score = val_metrics["coarse_plus_flow"]["pos_mae_mm"] + 0.1 * val_metrics["coarse_plus_flow"]["rot_mean_deg"]
        writer.add_scalar("epoch/model_selection_score", score, ep)

        if score < best_val:
            best_val = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "x_mean": scaler.x_mean.cpu(),
                    "x_std": scaler.x_std.cpu(),
                    "y_mean": scaler.y_mean.cpu(),
                    "y_std": scaler.y_std.cpu(),
                    "coarse_ckpt": coarse_ckpt_path,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            writer.add_scalar("epoch/best_score", best_val, ep)

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model"])

    final_metrics = evaluate(
        model=model,
        loader=va_loader,
        device=device,
        supervision=supervision,
        ref_frame=ref_frame,
        scaler=scaler,
        steps=infer_steps,
        n_samples=infer_n_samples,
        alpha=infer_alpha,
    )

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    m = cfg.get("model", {})
    writer.add_hparams(
        {
            "lr": float(cfg["train"].get("lr", 1e-4)),
            "weight_decay": float(cfg["train"].get("weight_decay", 1e-4)),
            "batch_size": int(cfg["train"]["batch_size"]),
            "epochs": epochs,
            "grad_clip_norm": grad_clip,
            "lambda_flow": lambda_flow,
            "infer_steps": infer_steps,
            "infer_n_samples": infer_n_samples,
            "flow_width": int(m.get("flow_width", 512)),
            "flow_depth": int(m.get("flow_depth", 4)),
            "flow_tfeat": int(m.get("flow_tfeat", 16)),
        },
        {
            "hparam/final_coarse_pos_mae_mm": final_metrics["coarse"]["pos_mae_mm"],
            "hparam/final_full_pos_mae_mm": final_metrics["coarse_plus_flow"]["pos_mae_mm"],
            "hparam/final_coarse_rot_mean_deg": final_metrics["coarse"]["rot_mean_deg"],
            "hparam/final_full_rot_mean_deg": final_metrics["coarse_plus_flow"]["rot_mean_deg"],
            "hparam/final_flow_residual_mse": final_metrics["flow"]["residual_mse"],
        },
    )

    writer.close()

    print("\n===== Final Result =====")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print("Saved best model to:", best_path)
    print("TensorBoard log dir:", tb_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.coarse_ckpt)

# python -m rigid_flow.train_mamba_with_flow --config configs/rigid_config.yaml --coarse_ckpt runs/mamba_coarse_only/best_model.pt

# tensorboard --logdir runs/mamba_with_flow_show/tensorboard