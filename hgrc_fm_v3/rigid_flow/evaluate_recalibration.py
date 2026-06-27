"""Evaluate cross-user and repeated-wearing performance with a frozen model.

Recalibration uses an initial target-session segment to fit only a positive
gain and bias for each of the three active fiber channels. The reconstruction
network and its training-set standardizer remain frozen.
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
import yaml

from rigid_flow import geometry as geom
from rigid_flow.infer_mamba_with_flow_csv import (
    SeqStandardizer,
    build_model_from_cfg,
    make_windows,
    parse_csv_no_header,
    preprocess_frames_from_csv,
    relative_to_world_if_needed,
    sample_flow_residual,
    world_to_relative_if_needed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output_dir", default="runs/generalization")
    parser.add_argument("--calibration_ratio", type=float, default=0.2)
    parser.add_argument("--calibration_steps", type=int, default=150)
    parser.add_argument("--calibration_lr", type=float, default=0.03)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def apply_adapter(x_std, log_gain, bias):
    gain = torch.exp(log_gain)
    active = x_std[..., 9:12] * gain + bias
    return torch.cat([x_std[..., :9], active, x_std[..., 12:13]], dim=-1)


def geodesic_angle(pred_r6, target_r6):
    pred_r = geom.r6d_to_matrix(pred_r6)
    target_r = geom.r6d_to_matrix(target_r6)
    delta = torch.einsum("bij,bjk->bik", pred_r.transpose(1, 2), target_r)
    trace = delta[:, 0, 0] + delta[:, 1, 1] + delta[:, 2, 2]
    cosine = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.arccos(cosine) * (180.0 / math.pi)


def fit_adapter(model, scaler, x_cal, y_cal, device, steps, lr):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    log_gain = torch.zeros(3, device=device, requires_grad=True)
    bias = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_gain, bias], lr=lr)
    batch_size = min(256, len(x_cal))

    x_cal = x_cal.to(device)
    y_cal = y_cal.to(device)
    for step in range(steps):
        indices = torch.randint(0, len(x_cal), (batch_size,), device=device)
        x_batch = apply_adapter(scaler.transform_x(x_cal[indices]), log_gain, bias)
        y_batch = y_cal[indices]

        prediction = model.coarse_only(x_batch)
        target = scaler.transform_y(y_batch)
        pose_loss = F.l1_loss(prediction[:, :3], target[:, :3])
        pose_loss = pose_loss + F.l1_loss(prediction[:, 3:9], target[:, 3:9])
        regularization = 0.01 * (log_gain.square().mean() + bias.square().mean())
        loss = pose_loss + regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in (0, steps - 1) or (step + 1) % 50 == 0:
            print(
                f"calibration step {step + 1:03d}/{steps}: "
                f"loss={loss.item():.5f}, "
                f"gain={torch.exp(log_gain).detach().cpu().tolist()}, "
                f"bias={bias.detach().cpu().tolist()}"
            )

    return log_gain.detach(), bias.detach()


@torch.no_grad()
def evaluate(model, scaler, x, y_target, y_world, device, cfg, log_gain=None, bias=None):
    model.eval()
    batch_size = 256
    steps = int(cfg.get("infer", {}).get("steps", 20))
    samples = int(cfg.get("infer", {}).get("n_samples", 1))
    alpha = float(cfg.get("infer", {}).get("alpha", 0.2))
    supervision = cfg["data"].get("supervision", "world")
    ref_frame = cfg["data"].get("ref_frame", "last")

    pred_world_all = []
    gt_world_all = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb = x[start:end].to(device)
        x_std = scaler.transform_x(xb)
        if log_gain is not None:
            x_std = apply_adapter(x_std, log_gain, bias)

        coarse = model.coarse_only(x_std)
        residual = sample_flow_residual(model, x_std, steps=steps, n_samples=samples)
        pred_raw = scaler.inverse_y(coarse + alpha * residual)
        pred_world = relative_to_world_if_needed(pred_raw, xb, supervision, ref_frame)

        pred_world_all.append(pred_world.cpu())
        gt_world_all.append(y_world[start:end].cpu())

    pred = torch.cat(pred_world_all)
    target = torch.cat(gt_world_all)
    pos_error = pred[:, :3] - target[:, :3]
    l2 = torch.linalg.norm(pos_error, dim=-1)
    rot = geodesic_angle(pred[:, 3:9], target[:, 3:9])
    return {
        "num_windows": int(len(pred)),
        "pos_mae_mm": float(pos_error.abs().mean().item()),
        "pos_l2_mm": float(l2.mean().item()),
        "pos_rmse_mm": float(torch.sqrt((l2.square()).mean()).item()),
        "rot_mean_deg": float(rot.mean().item()),
        "rot_med_deg": float(rot.median().item()),
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model_cfg = checkpoint.get("cfg", cfg)
    model = build_model_from_cfg(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)

    scaler = SeqStandardizer()
    scaler.x_mean = checkpoint["x_mean"].float()
    scaler.x_std = checkpoint["x_std"].float()
    scaler.y_mean = checkpoint["y_mean"].float()
    scaler.y_std = checkpoint["y_std"].float()

    data_cfg = model_cfg["data"]
    raw = parse_csv_no_header(args.input_csv)
    split = int(len(raw) * args.calibration_ratio)
    window_size = int(data_cfg["window_size"])
    if split < window_size or len(raw) - split < window_size:
        raise ValueError("Calibration or test segment is shorter than one window.")

    def prepare(segment):
        back, tip = preprocess_frames_from_csv(
            segment,
            pos_unit=data_cfg.get("pos_unit", "mm"),
            sensor_scale=float(data_cfg.get("sensor_scale", 1024.0)),
        )
        x, y_world, _ = make_windows(
            back, tip, window_size=window_size, ref_frame=data_cfg.get("ref_frame", "last")
        )
        y_target = world_to_relative_if_needed(
            y_world, x, data_cfg.get("supervision", "world"), data_cfg.get("ref_frame", "last")
        )
        return x, y_target, y_world

    x_cal, y_cal, _ = prepare(raw[:split])
    x_test, y_test, y_test_world = prepare(raw[split:])

    print(f"{args.name}: frames={len(raw)}, calibration={split}, test={len(raw) - split}")
    without = evaluate(model, scaler, x_test, y_test, y_test_world, device, model_cfg)
    log_gain, bias = fit_adapter(
        model,
        scaler,
        x_cal,
        y_cal,
        device,
        steps=args.calibration_steps,
        lr=args.calibration_lr,
    )
    with_calibration = evaluate(
        model, scaler, x_test, y_test, y_test_world, device, model_cfg, log_gain, bias
    )

    result = {
        "name": args.name,
        "protocol": {
            "calibration_ratio": args.calibration_ratio,
            "calibration_frames": split,
            "test_frames": len(raw) - split,
            "network_frozen": True,
            "calibrated_parameters": "per-channel positive gain and bias for three active fiber channels",
        },
        "adapter": {
            "gain": torch.exp(log_gain).cpu().tolist(),
            "bias": bias.cpu().tolist(),
        },
        "without_recalibration": without,
        "with_recalibration": with_calibration,
    }
    output = os.path.join(args.output_dir, f"{args.name}.json")
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
