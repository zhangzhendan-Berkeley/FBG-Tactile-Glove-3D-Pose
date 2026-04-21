# -*- coding: utf-8 -*-
"""
把 clean_glove_one_row_per_frame_4000_45400.csv
转换成师兄原本 rigid_flow/data.py 可直接读取的 txt 格式。

输入 CSV 每行:
frame_idx, ch1, ch2, ch3, ch4,
back_lt_id, back_lt_x, back_lt_y, back_lt_z,
back_rt_id, back_rt_x, back_rt_y, back_rt_z,
back_rb_id, back_rb_x, back_rb_y, back_rb_z,
back_lb_id, back_lb_x, back_lb_y, back_lb_z,
tip_lt_id,  tip_lt_x,  tip_lt_y,  tip_lt_z,
tip_rt_id,  tip_rt_x,  tip_rt_y,  tip_rt_z,
tip_rb_id,  tip_rb_x,  tip_rb_y,  tip_rb_z,
tip_lb_id,  tip_lb_x,  tip_lb_y,  tip_lb_z

输出 TXT 每行:
id_back, back_x, back_y, back_z, back_qx, back_qy, back_qz, back_qw,
id_tip,  tip_x,  tip_y,  tip_z,  tip_qx,  tip_qy,  tip_qz,  tip_qw,
s0, s1, s2, s3

注意：
- 你的输入 CSV 坐标语义是 yzx
- 师兄原 data.py 假设输入 txt 是 xyz，再内部做 xyz -> yzx
- 所以这里导出的 txt 会把 yzx 逆变换回 xyz
"""

import os
import math
import argparse
import numpy as np
import pandas as pd


# -----------------------------
# 坐标语义变换
# -----------------------------
# 师兄 data.py 里是 vec_xyz_to_yzx: [x, y, z] -> [y, z, x]
# 这里要做反变换： [y, z, x] -> [x, y, z]
def vec_yzx_to_xyz(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([v[2], v[0], v[1]], dtype=np.float64)


def get_xyz_to_yzx_perm_matrix():
    # v_yzx = P @ v_xyz
    return np.array([
        [0.0, 1.0, 0.0],  # y
        [0.0, 0.0, 1.0],  # z
        [1.0, 0.0, 0.0],  # x
    ], dtype=np.float64)


def rot_yzx_to_xyz(R_yzx):
    """
    已知旋转矩阵是在 yzx 坐标语义下表示的，
    转回 xyz 语义，供师兄原版 data.py 再映射回 yzx。
    关系：
        R_yzx = P @ R_xyz @ P^T
    所以：
        R_xyz = P^T @ R_yzx @ P
    """
    P = get_xyz_to_yzx_perm_matrix()
    return P.T @ R_yzx @ P


# -----------------------------
# 旋转矩阵 / 四元数
# -----------------------------
def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def orthonormalize_axes(x_axis, y_hint):
    """
    给定 x 轴和一个 y 的提示方向，构造右手正交基
    """
    x = normalize(x_axis)
    if x is None:
        return None

    z = np.cross(x, y_hint)
    z = normalize(z)
    if z is None:
        return None

    y = np.cross(z, x)
    y = normalize(y)
    if y is None:
        return None

    R = np.stack([x, y, z], axis=1)  # 列向量为基向量
    return R


def matrix_to_quat_xyzw(R):
    """
    3x3 rotation matrix -> quaternion [x, y, z, w]
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)

    # 可选：让 qw >= 0，减少符号跳变
    if q[3] < 0:
        q = -q
    return q


# -----------------------------
# 用四角点恢复刚体 pose
# -----------------------------
def rigid_pose_from_four_markers(lt, rt, rb, lb):
    """
    输入四个角点（当前 CSV 的 yzx 语义坐标）：
        lt, rt, rb, lb : shape [3]

    构造一个统一的刚体坐标系：
    - x 轴：左 -> 右
    - y 轴：下 -> 上
    - z 轴：x × y
    - 平移：四点质心

    返回：
        center_yzx: [3]
        R_yzx: [3,3]
    """
    lt = np.asarray(lt, dtype=np.float64)
    rt = np.asarray(rt, dtype=np.float64)
    rb = np.asarray(rb, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)

    center = (lt + rt + rb + lb) / 4.0

    # 左右方向：右边两点中心 - 左边两点中心
    x_axis = ((rt + rb) * 0.5) - ((lt + lb) * 0.5)

    # 上下方向：上边两点中心 - 下边两点中心
    y_hint = ((lt + rt) * 0.5) - ((lb + rb) * 0.5)

    R = orthonormalize_axes(x_axis, y_hint)
    if R is None:
        return None, None

    # 修正 det，确保为真正旋转矩阵
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    return center, R


# -----------------------------
# 读取四个 marker 点
# -----------------------------
def get_body_markers_from_row(row, prefix):
    """
    prefix: 'back' 或 'tip'
    返回 lt, rt, rb, lb, ids
    """
    lt = np.array([row[f"{prefix}_lt_x"], row[f"{prefix}_lt_y"], row[f"{prefix}_lt_z"]], dtype=np.float64)
    rt = np.array([row[f"{prefix}_rt_x"], row[f"{prefix}_rt_y"], row[f"{prefix}_rt_z"]], dtype=np.float64)
    rb = np.array([row[f"{prefix}_rb_x"], row[f"{prefix}_rb_y"], row[f"{prefix}_rb_z"]], dtype=np.float64)
    lb = np.array([row[f"{prefix}_lb_x"], row[f"{prefix}_lb_y"], row[f"{prefix}_lb_z"]], dtype=np.float64)

    ids = [
        row[f"{prefix}_lt_id"],
        row[f"{prefix}_rt_id"],
        row[f"{prefix}_rb_id"],
        row[f"{prefix}_lb_id"],
    ]
    return lt, rt, rb, lb, ids


def is_valid_point(p):
    return np.all(np.isfinite(p))


def row_is_valid(row):
    required = ["ch1", "ch2", "ch3", "ch4"]
    for c in required:
        if c not in row or not np.isfinite(row[c]):
            return False
    return True


# -----------------------------
# 主转换逻辑
# -----------------------------
def convert_csv_to_pose_txt(csv_path, txt_path):
    df = pd.read_csv(csv_path)

    needed_cols = [
        "frame_idx", "ch1", "ch2", "ch3", "ch4",
        "back_lt_id", "back_lt_x", "back_lt_y", "back_lt_z",
        "back_rt_id", "back_rt_x", "back_rt_y", "back_rt_z",
        "back_rb_id", "back_rb_x", "back_rb_y", "back_rb_z",
        "back_lb_id", "back_lb_x", "back_lb_y", "back_lb_z",
        "tip_lt_id", "tip_lt_x", "tip_lt_y", "tip_lt_z",
        "tip_rt_id", "tip_rt_x", "tip_rt_y", "tip_rt_z",
        "tip_rb_id", "tip_rb_x", "tip_rb_y", "tip_rb_z",
        "tip_lb_id", "tip_lb_x", "tip_lb_y", "tip_lb_z",
    ]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")

    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)

    kept = 0
    skipped = 0

    with open(txt_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if not row_is_valid(row):
                skipped += 1
                continue

            back_lt, back_rt, back_rb, back_lb, back_ids = get_body_markers_from_row(row, "back")
            tip_lt, tip_rt, tip_rb, tip_lb, tip_ids = get_body_markers_from_row(row, "tip")

            pts = [back_lt, back_rt, back_rb, back_lb, tip_lt, tip_rt, tip_rb, tip_lb]
            if not all(is_valid_point(p) for p in pts):
                skipped += 1
                continue

            # 当前 CSV 是 yzx 语义
            back_center_yzx, back_R_yzx = rigid_pose_from_four_markers(back_lt, back_rt, back_rb, back_lb)
            tip_center_yzx, tip_R_yzx = rigid_pose_from_four_markers(tip_lt, tip_rt, tip_rb, tip_lb)

            if back_R_yzx is None or tip_R_yzx is None:
                skipped += 1
                continue

            # 为了兼容师兄原 data.py，这里导出成 xyz 语义
            back_p_xyz = vec_yzx_to_xyz(back_center_yzx)
            tip_p_xyz = vec_yzx_to_xyz(tip_center_yzx)

            back_R_xyz = rot_yzx_to_xyz(back_R_yzx)
            tip_R_xyz = rot_yzx_to_xyz(tip_R_yzx)

            back_q_xyzw = matrix_to_quat_xyzw(back_R_xyz)
            tip_q_xyzw = matrix_to_quat_xyzw(tip_R_xyz)

            s0 = float(row["ch1"])
            s1 = float(row["ch2"])
            s2 = float(row["ch3"])
            s3 = float(row["ch4"])

            # 这里 rigid body id 直接按师兄格式写 1 / 2
            vals = [
                1,
                back_p_xyz[0], back_p_xyz[1], back_p_xyz[2],
                back_q_xyzw[0], back_q_xyzw[1], back_q_xyzw[2], back_q_xyzw[3],

                2,
                tip_p_xyz[0], tip_p_xyz[1], tip_p_xyz[2],
                tip_q_xyzw[0], tip_q_xyzw[1], tip_q_xyzw[2], tip_q_xyzw[3],

                s0, s1, s2, s3
            ]

            line = ",".join(f"{v:.8f}" if isinstance(v, float) or isinstance(v, np.floating) else str(v) for v in vals)
            f.write(line + "\n")
            kept += 1

    print(f"转换完成: {txt_path}")
    print(f"保留帧数: {kept}")
    print(f"跳过帧数: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="clean_glove_one_row_per_frame_4000_45400.csv",
        help="输入 CSV 路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="glove_pose_for_senior_data_py.txt",
        help="输出 txt 路径"
    )
    args = parser.parse_args()

    convert_csv_to_pose_txt(args.input, args.output)


if __name__ == "__main__":
    main()