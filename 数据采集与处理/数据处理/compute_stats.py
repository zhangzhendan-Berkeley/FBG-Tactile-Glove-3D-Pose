# rigid_flow/compute_stats.py
# -*- coding: utf-8 -*-
"""
Compute tip position statistics (mean/std in mm, world frame) for training normalization.

Usage:
  python -m rigid_flow.compute_stats --config configs/rigid_config.yaml

It reads `data.train_files` from the YAML, builds a RigidSeqDataset, and scans all raw frames
(ignoring windowing). We always use WORLD tip position in mm (yzx) for statistics,
regardless of whether training supervision uses 'world' or 'relative'.
"""

import os, yaml, argparse, numpy as np

from data import RigidSeqDataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    stats_path = data_cfg.get("stats_path", "runs/pos_stats_mm.yaml")
    pos_unit = data_cfg.get("pos_unit", "mm")

    # 构建数据集：我们只需要把所有文件的“逐帧原始数据”读进来；
    # window_size/stride/sensor_scale/supervision等不会影响 frames 的原始值。
    ds = RigidSeqDataset(
        files=data_cfg["train_files"],
        schema_file=data_cfg.get("schema_file"),
        window_size=data_cfg.get("window_size", 64),
        window_stride=data_cfg.get("window_stride", 1),
        sensor_scale=data_cfg.get("sensor_scale", 1024.0),
        stats_path=None,                  # 不依赖已有统计
        mode="train",
        pos_unit=pos_unit,
        supervision=data_cfg.get("supervision", "world"),
        ref_frame=data_cfg.get("ref_frame", "last"),
    )

    # 新 data.py 中，逐帧数据在 ds.frames（list of dict）里
    # 每个元素含有：
    #   "tip_p": tip 世界系位置 (3, mm, yzx)
    #   以及 back/tip 的姿态、sensor 等
    if not hasattr(ds, "frames") or len(ds.frames) == 0:
        raise RuntimeError("Dataset has no frames; please check your data files and config.")

    tips = []
    for fr in ds.frames:
        tp = fr.get("tip_p", None)
        if tp is None:
            # 兜底：尝试旧字段（理论上不会走到这里）
            raise RuntimeError("Frame missing 'tip_p'; ensure you're using the provided data.py.")
        tips.append(tp)

    tips = np.asarray(tips, dtype=np.float64)  # [N,3], mm
    mean_mm = tips.mean(axis=0)
    std_mm  = tips.std(axis=0, ddof=0)  # population std

    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    out = {
        "tip_pos_mm_mean": [float(x) for x in mean_mm.tolist()],
        "tip_pos_mm_std":  [float(max(x, 1e-6)) for x in std_mm.tolist()],  # clamp to avoid zero
        "count": int(tips.shape[0]),
        "note": "Statistics computed in WORLD frame (yzx), unit mm, from training files."
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Saved tip position stats (mm) to: {stats_path}")
    print(f"      mean_mm = {out['tip_pos_mm_mean']}")
    print(f"      std_mm  = {out['tip_pos_mm_std']}")
    print(f"      count   = {out['count']}")

if __name__ == "__main__":
    main()
