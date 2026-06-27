# -*- coding: utf-8 -*-
"""
测试 rigid_flow/data.py / geometry.py 中的姿态处理链路：

原始四元数
-> quat_to_matrix
-> (可选 xyz->yzx 映射)
-> rot_to_6d
-> r6d_to_matrix

检查：
1. quat -> matrix -> rot6d -> matrix 是否自洽
2. xyz->yzx 映射后是否自洽
3. 哪些帧误差最大
"""

import os
import math
import argparse
import numpy as np
import torch

from rigid_flow import geometry as geom


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="原始 txt 文件路径")
    ap.add_argument("--max_frames", type=int, default=-1, help="最多检查多少帧，-1 表示全部")
    ap.add_argument("--print_topk", type=int, default=10, help="打印误差最大的前 k 帧")
    return ap.parse_args()


def geodesic_angle_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """
    Ra, Rb: [B,3,3]
    return: [B]，单位 degree
    """
    M = torch.einsum("bij,bjk->bik", Ra.transpose(1, 2), Rb)
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    cos = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    ang = torch.arccos(cos)
    return ang * (180.0 / math.pi)


def load_raw_file(path, max_frames=-1):
    """
    读取原始文件，每行格式：
    rb1_id, rb1_x, rb1_y, rb1_z, rb1_qx, rb1_qy, rb1_qz, rb1_qw,
    rb2_id, rb2_x, rb2_y, rb2_z, rb2_qx, rb2_qy, rb2_qz, rb2_qw,
    s0, s1, s2, s3
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [x.strip() for x in line.split(",") if x.strip() != ""]
            try:
                nums = list(map(float, vals))
            except Exception:
                continue
            if len(nums) != 20:
                continue
            rows.append(nums)
            if max_frames > 0 and len(rows) >= max_frames:
                break

    if len(rows) == 0:
        raise RuntimeError(f"没有从文件中读到有效帧: {path}")

    arr = np.asarray(rows, dtype=np.float64)
    return arr


def summarize_errors(name, deg_err: torch.Tensor):
    x = deg_err.detach().cpu().numpy()
    print(f"\n===== {name} =====")
    print(f"count      = {len(x)}")
    print(f"mean deg   = {x.mean():.8f}")
    print(f"median deg = {np.median(x):.8f}")
    print(f"max deg    = {x.max():.8f}")
    print(f"min deg    = {x.min():.8f}")
    print(f"p95 deg    = {np.percentile(x, 95):.8f}")
    print(f"p99 deg    = {np.percentile(x, 99):.8f}")


def print_topk_frames(name, deg_err: torch.Tensor, topk: int):
    x = deg_err.detach().cpu().numpy()
    idx = np.argsort(-x)[:topk]
    print(f"\n{name} 误差最大的前 {len(idx)} 帧：")
    for i in idx:
        print(f"  frame={i:6d} | err_deg={x[i]:.8f}")


def main():
    args = parse_args()

    raw = load_raw_file(args.input, args.max_frames)
    N = len(raw)
    print(f"Loaded {N} valid frames from: {args.input}")

    # 提取 back / tip 四元数
    back_q = torch.tensor(raw[:, 4:8], dtype=torch.float32)   # [N,4]
    tip_q  = torch.tensor(raw[:, 12:16], dtype=torch.float32) # [N,4]

    # ---------------------------------------------------
    # A. 最基础链路：原始四元数 -> 矩阵 -> rot6d -> 矩阵
    # ---------------------------------------------------
    back_R = geom.quat_to_matrix(back_q)          # [N,3,3]
    tip_R  = geom.quat_to_matrix(tip_q)

    back_r6 = geom.rot_to_6d(back_R)              # [N,6]
    tip_r6  = geom.rot_to_6d(tip_R)

    back_R_rec = geom.r6d_to_matrix(back_r6)      # [N,3,3]
    tip_R_rec  = geom.r6d_to_matrix(tip_r6)

    back_err_A = geodesic_angle_deg(back_R, back_R_rec)
    tip_err_A  = geodesic_angle_deg(tip_R, tip_R_rec)

    summarize_errors("A. back: quat -> R -> rot6d -> R", back_err_A)
    summarize_errors("A. tip : quat -> R -> rot6d -> R", tip_err_A)

    print_topk_frames("A. back", back_err_A, args.print_topk)
    print_topk_frames("A. tip ", tip_err_A, args.print_topk)

    # ---------------------------------------------------
    # B. 模拟 data.py 中 xyz -> yzx 映射后的链路
    # quat -> R -> rot_xyz_to_yzx -> rot6d -> R
    # ---------------------------------------------------
    back_R_yzx = geom.rot_xyz_to_yzx(back_R)
    tip_R_yzx  = geom.rot_xyz_to_yzx(tip_R)

    back_r6_yzx = geom.rot_to_6d(back_R_yzx)
    tip_r6_yzx  = geom.rot_to_6d(tip_R_yzx)

    back_R_yzx_rec = geom.r6d_to_matrix(back_r6_yzx)
    tip_R_yzx_rec  = geom.r6d_to_matrix(tip_r6_yzx)

    back_err_B = geodesic_angle_deg(back_R_yzx, back_R_yzx_rec)
    tip_err_B  = geodesic_angle_deg(tip_R_yzx, tip_R_yzx_rec)

    summarize_errors("B. back: quat -> R -> yzx-map -> rot6d -> R", back_err_B)
    summarize_errors("B. tip : quat -> R -> yzx-map -> rot6d -> R", tip_err_B)

    print_topk_frames("B. back", back_err_B, args.print_topk)
    print_topk_frames("B. tip ", tip_err_B, args.print_topk)

    # ---------------------------------------------------
    # C. 检查 quat_xyz_to_yzx 是否和 rot_xyz_to_yzx 一致
    # quat -> quat_xyz_to_yzx -> R
    # 和
    # quat -> R -> rot_xyz_to_yzx
    # ---------------------------------------------------
    back_q_yzx = geom.quat_xyz_to_yzx(back_q)
    tip_q_yzx  = geom.quat_xyz_to_yzx(tip_q)

    back_R_from_qmap = geom.quat_to_matrix(back_q_yzx)
    tip_R_from_qmap  = geom.quat_to_matrix(tip_q_yzx)

    back_err_C = geodesic_angle_deg(back_R_yzx, back_R_from_qmap)
    tip_err_C  = geodesic_angle_deg(tip_R_yzx, tip_R_from_qmap)

    summarize_errors("C. back: rot_xyz_to_yzx(R) vs quat_xyz_to_yzx(q)", back_err_C)
    summarize_errors("C. tip : rot_xyz_to_yzx(R) vs quat_xyz_to_yzx(q)", tip_err_C)

    print_topk_frames("C. back", back_err_C, args.print_topk)
    print_topk_frames("C. tip ", tip_err_C, args.print_topk)

    # ---------------------------------------------------
    # D. 打印几帧详细信息，方便人工看
    # ---------------------------------------------------
    print("\n===== 随机/关键帧详细检查 =====")
    inspect_ids = [0, min(1, N-1), min(2, N-1), min(10, N-1), min(N-1, 50)]
    inspect_ids = sorted(set([i for i in inspect_ids if 0 <= i < N]))

    for i in inspect_ids:
        print(f"\n--- frame {i} ---")

        print("back_q =", back_q[i].tolist())
        print("tip_q  =", tip_q[i].tolist())

        print("back_R =")
        print(back_R[i].numpy())
        print("back_r6 =")
        print(back_r6[i].numpy())
        print("back_R_rec =")
        print(back_R_rec[i].numpy())
        print(f"back_err_A_deg = {back_err_A[i].item():.8f}")

        print("tip_R =")
        print(tip_R[i].numpy())
        print("tip_r6 =")
        print(tip_r6[i].numpy())
        print("tip_R_rec =")
        print(tip_R_rec[i].numpy())
        print(f"tip_err_A_deg = {tip_err_A[i].item():.8f}")

    # ---------------------------------------------------
    # E. 给出诊断结论
    # ---------------------------------------------------
    maxA = max(float(back_err_A.max().item()), float(tip_err_A.max().item()))
    maxB = max(float(back_err_B.max().item()), float(tip_err_B.max().item()))
    maxC = max(float(back_err_C.max().item()), float(tip_err_C.max().item()))

    print("\n===== 诊断建议 =====")
    if maxA < 1e-3:
        print("A 链路基本正常：quat -> R -> rot6d -> R 自洽。")
    else:
        print("A 链路不正常：问题很可能在 rot_to_6d 或 r6d_to_matrix。")

    if maxB < 1e-3:
        print("B 链路基本正常：yzx 映射后的 rot6d 编解码自洽。")
    else:
        print("B 链路不正常：问题可能出在 rot_to_6d / r6d_to_matrix，或 yzx 映射后表示不一致。")

    if maxC < 1e-3:
        print("C 链路基本正常：quat_xyz_to_yzx 与 rot_xyz_to_yzx 是一致的。")
    else:
        print("C 链路不正常：问题可能在 matrix_to_quat / quat_xyz_to_yzx。")


if __name__ == "__main__":
    main()