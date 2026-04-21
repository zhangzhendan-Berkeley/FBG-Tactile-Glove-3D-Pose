# rigid_flow/compute_stats.py
# -*- coding: utf-8 -*-
"""
Compute tip position statistics (mean/std in mm, world frame) for training normalization.

Usage:
  python -m rigid_flow.compute_stats --config configs/rigid_config.yaml
"""

import os
import yaml
import argparse
import numpy as np

from .data import RigidSeqDataset


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

    ds = RigidSeqDataset(
        files=data_cfg["train_files"],
        schema_file=data_cfg.get("schema_file"),
        window_size=data_cfg.get("window_size", 64),
        window_stride=data_cfg.get("window_stride", 1),
        sensor_scale=data_cfg.get("sensor_scale", 1024.0),
        stats_path=None,
        mode="train",
        pos_unit=pos_unit,
        supervision=data_cfg.get("supervision", "world"),
        ref_frame=data_cfg.get("ref_frame", "last"),
    )

    tips = []

    if hasattr(ds, "frames") and len(ds.frames) > 0:
        for fr in ds.frames:
            tp = fr.get("tip_p", None)
            if tp is None:
                tp = fr.get("tip_p_mm", None)
            if tp is None:
                raise RuntimeError("Frame missing 'tip_p' / 'tip_p_mm'.")
            if hasattr(tp, "detach"):
                tp = tp.detach().cpu().numpy()
            tips.append(tp)

    elif hasattr(ds, "frames_by_file") and len(ds.frames_by_file) > 0:
        for arr in ds.frames_by_file:
            for fr in arr:
                tp = fr.get("tip_p", None)
                if tp is None:
                    tp = fr.get("tip_p_mm", None)
                if tp is None:
                    raise RuntimeError("Frame missing 'tip_p' / 'tip_p_mm'.")
                if hasattr(tp, "detach"):
                    tp = tp.detach().cpu().numpy()
                tips.append(tp)

    else:
        raise RuntimeError("Dataset has no raw frames; please check your data files and config.")

    if len(tips) == 0:
        raise RuntimeError("No valid tip positions found in dataset.")

    tips = np.asarray(tips, dtype=np.float64)  # [N,3], mm
    mean_mm = tips.mean(axis=0)
    std_mm = tips.std(axis=0, ddof=0)

    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    out = {
        "tip_pos_mm_mean": [float(x) for x in mean_mm.tolist()],
        "tip_pos_mm_std": [float(max(x, 1e-6)) for x in std_mm.tolist()],
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