# convert_and_split_csv_to_mamba.py
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

# =========================
# 直接在这里改
# =========================
INPUT_CSV   = r"clean_frames_with_angle.csv"
OUT_TRAIN   = r"data/train_mamba.txt"
OUT_TEST    = r"data/test_mamba.txt"

TRAIN_RATIO = 0.9     # 前 90% 作为训练集
RB_QW_FILL  = 1.0     # quat 的 w，0.0 或 1.0 都行（推荐 1.0 更稳）

# =========================
# CSV 列名映射（按你给的表头）
# =========================
A = ("x_p102782", "y_p102782", "z_p102782")  # 第一个点
B = ("x_p102800", "y_p102800", "z_p102800")  # 第二个点
C = ("x_p102797", "y_p102797", "z_p102797")  # 第三个点
D = ("x_p102798", "y_p102798", "z_p102798")  # 第四个点
V0_COL = "v1"                                # 只用第一个电压

RB1_ID = 1
RB2_ID = 2


def fnum(x):
    x = float(x)
    s = f"{x:.8f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def convert_row(row):
    """CSV 行 -> mamba 一行（list[float/int]）"""
    ax, ay, az = (float(row[A[0]]), float(row[A[1]]), float(row[A[2]]))
    bx, by, bz = (float(row[B[0]]), float(row[B[1]]), float(row[B[2]]))
    cx, cy, cz = (float(row[C[0]]), float(row[C[1]]), float(row[C[2]]))
    dx, dy, dz = (float(row[D[0]]), float(row[D[1]]), float(row[D[2]]))
    v0 = float(row[V0_COL])

    sensors = [v0, v0, v0, v0]

    return [
        RB1_ID,
        ax, ay, az,
        bx, by, bz, RB_QW_FILL,
        RB2_ID,
        cx, cy, cz,
        dx, dy, dz, RB_QW_FILL,
        *sensors
    ]


def write_rows(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            parts = []
            for i, x in enumerate(row):
                if i in (0, 8):  # 两个 id
                    parts.append(str(int(x)))
                else:
                    parts.append(fnum(x))
            f.write(",".join(parts) + "\n")


def main():
    in_path = Path(INPUT_CSV)
    rows_all = []

    with in_path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)

        # 检查列
        needed = list(A + B + C + D) + [V0_COL]
        missing = [k for k in needed if k not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"CSV 缺少列：{missing}")

        for row in reader:
            try:
                rows_all.append(convert_row(row))
            except Exception:
                continue

    n = len(rows_all)
    n_train = int(n * TRAIN_RATIO)

    train_rows = rows_all[:n_train]
    test_rows  = rows_all[n_train:]

    write_rows(train_rows, Path(OUT_TRAIN))
    write_rows(test_rows,  Path(OUT_TEST))

    print("=== Done ===")
    print(f"total frames : {n}")
    print(f"train frames : {len(train_rows)}")
    print(f"test frames  : {len(test_rows)}")
    print(f"train file   : {OUT_TRAIN}")
    print(f"test file    : {OUT_TEST}")


if __name__ == "__main__":
    main()
