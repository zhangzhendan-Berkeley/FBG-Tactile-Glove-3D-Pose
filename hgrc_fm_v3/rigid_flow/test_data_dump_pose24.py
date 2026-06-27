# rigid_flow/test_data_dump_pose24.py
# -*- coding: utf-8 -*-

import os
import csv
import yaml
import argparse
import numpy as np
import torch

from .data import RigidSeqDataset
from rigid_flow import geometry as geom


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--out_dir", type=str, default="runs/data_dump_pose24")
    ap.add_argument("--max_samples", type=int, default=-1)
    return ap.parse_args()


def build_dataset(cfg, split: str):
    if split == "train":
        files = cfg["data"]["train_files"]
        mode = "train"
    else:
        files = cfg["data"]["test_files"]
        mode = "test"

    ds = RigidSeqDataset(
        files=files,
        schema_file=cfg["data"].get("schema_file"),
        window_size=cfg["data"]["window_size"],
        window_stride=cfg["data"]["window_stride"],
        sensor_scale=cfg["data"].get("sensor_scale", 1024.0),
        stats_path=cfg["data"].get("stats_path"),
        mode=mode,
        pos_unit=cfg["data"].get("pos_unit", "mm"),
        supervision=cfg["data"].get("supervision", "world"),
        ref_frame=cfg["data"].get("ref_frame", "last"),
        augment=None,   # 测试 data.py 时不要开增广
    )
    return ds


def sample_to_pose24(ds: RigidSeqDataset, i: int):
    """
    把一个 sample 还原成：
    [back_id, back_p3, back_r6_6, tip_id, tip_p3, tip_r6_6, sensor4]
    共 24 列
    """
    item = ds[i]
    back_seq = item["back_seq"].clone()     # [T,13]
    y9 = item["y9_target"].clone()          # [9]

    T = back_seq.shape[0]
    idx = (T // 2) if (ds.ref_frame == "center") else (T - 1)

    # 参考帧的 back
    back_p = back_seq[idx, 0:3]            # [3]
    back_r6 = back_seq[idx, 3:9]           # [6]
    sensors = back_seq[idx, 9:13]          # [4]，已经是 data.py 处理后的输入值

    if ds.supervision == "world":
        tip_p = y9[0:3]
        tip_r6 = y9[3:9]
    else:
        # y9 是 relative，要还原回 world
        y_pos_rel = y9[0:3]                        # [3]
        y_r6_rel = y9[3:9]                         # [6]

        R_back = geom.r6d_to_matrix(back_r6.unsqueeze(0))[0]    # [3,3]
        R_rel = geom.r6d_to_matrix(y_r6_rel.unsqueeze(0))[0]    # [3,3]

        tip_p = torch.einsum("ij,j->i", R_back, y_pos_rel) + back_p
        R_tip = torch.einsum("ij,jk->ik", R_back, R_rel)
        tip_r6 = geom.rot_to_6d(R_tip)

    row = torch.cat([
        torch.tensor([1.0], dtype=torch.float32),
        back_p.float(),
        back_r6.float(),
        torch.tensor([2.0], dtype=torch.float32),
        tip_p.float(),
        tip_r6.float(),
        sensors.float(),
    ], dim=0)

    return row.numpy()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = build_dataset(cfg, args.split)

    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    n_total = len(ds)
    n_dump = n_total if args.max_samples < 0 else min(n_total, args.max_samples)

    print(f"split          : {args.split}")
    print(f"dataset len    : {n_total}")
    print(f"dump samples   : {n_dump}")
    print(f"supervision    : {ds.supervision}")
    print(f"ref_frame      : {ds.ref_frame}")

    rows = []
    meta_rows = []

    for i in range(n_dump):
        row = sample_to_pose24(ds, i)
        rows.append(row)

        if hasattr(ds, "index"):
            fid, start = ds.index[i]
        else:
            fid, start = -1, -1
        meta_rows.append([i, fid, start])

    arr = np.stack(rows, axis=0)   # [N,24]

    # 1) 无表头 txt，最适合直接喂给你现有可视化脚本
    txt_path = os.path.join(out_dir, "pose24_no_header.txt")
    np.savetxt(txt_path, arr, fmt="%.8f", delimiter=",")

    # 2) 有表头 csv，方便检查
    csv_path = os.path.join(out_dir, "pose24_with_header.csv")
    header = [
        "back_id",
        "back_p0", "back_p1", "back_p2",
        "back_r6_0", "back_r6_1", "back_r6_2", "back_r6_3", "back_r6_4", "back_r6_5",
        "tip_id",
        "tip_p0", "tip_p1", "tip_p2",
        "tip_r6_0", "tip_r6_1", "tip_r6_2", "tip_r6_3", "tip_r6_4", "tip_r6_5",
        "sensor0", "sensor1", "sensor2", "sensor3"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(arr.tolist())

    # 3) sample 对应关系
    meta_path = os.path.join(out_dir, "sample_meta.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "fid", "start"])
        writer.writerows(meta_rows)

    print("\nSaved:")
    print(" ", txt_path)
    print(" ", csv_path)
    print(" ", meta_path)

    print("\nFirst row preview:")
    print(",".join([f"{x:.8f}" for x in arr[0]]))


if __name__ == "__main__":
    main()

# python -m rigid_flow.test_data_dump_pose24 --config configs/rigid_config.yaml --split train