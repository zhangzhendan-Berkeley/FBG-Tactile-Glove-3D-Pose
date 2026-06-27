import numpy as np
import pandas as pd

FILE_PATH = "sync_data-260121.txt"

# 你最终可以手动固定目标点；现在先自动
# TARGET_POINT_IDS = [101815, 101814, 101813, 101812]
TARGET_POINT_IDS = [102782, 102800, 102797, 102798]
SENSOR_COLS = ["v1", "v2", "v3", "v4"]

# 传感器异常
DROP_IF_ALL_SENSOR_ZERO = True
DROP_IF_V1_ZERO = True

# 离群：帧间差分 robust z
POS_DIFF_MAD_Z = 8.0
SENSOR_DIFF_MAD_Z = 10.0

OUT_SUMMARY_POINTS = "summary_points.csv"
OUT_FRAMES_CSV = "clean_frames.csv"
OUT_LONG_CSV = "clean_long.csv"


def parse_line(line: str):
    s = line.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    if len(parts) < 7:
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None

    if len(nums) == 7:
        nums.append(np.nan)
    if len(nums) > 8:
        nums = nums[:8]
    if len(nums) != 8:
        return None

    pid = int(nums[0])
    x, y, z = nums[1], nums[2], nums[3]
    v1, v2, v3, v4 = nums[4], nums[5], nums[6], nums[7]
    return pid, x, y, z, v1, v2, v3, v4


def assign_frames_by_pid_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    粗分帧：顺序扫描，当前帧内 point_id 不应重复；
    一旦重复就开新帧。
    """
    frame_idx = 0
    seen = set()
    out = []

    for row in df.itertuples(index=False):
        pid = row.point_id
        if pid in seen:
            frame_idx += 1
            seen = set()
        seen.add(pid)
        out.append(frame_idx)

    df = df.copy()
    df["frame_idx"] = out
    return df


def robust_zscore_by_mad(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def summarize_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    给每个点做统计：
    - appear_frames: 出现过的帧数
    - appear_ratio: 出现率 = appear_frames / total_frames
    - pos_var_sum: xyz 方差和（越小越“固定”）
    - pos_std_sum: xyz 标准差和（同上，更直观）
    """
    total_frames = df["frame_idx"].nunique()

    d = df.drop_duplicates(subset=["frame_idx", "point_id"], keep="first")

    appear = d.groupby("point_id")["frame_idx"].nunique().rename("appear_frames")
    var_xyz = d.groupby("point_id")[["x", "y", "z"]].var().rename(columns=lambda c: f"var_{c}")
    std_xyz = d.groupby("point_id")[["x", "y", "z"]].std().rename(columns=lambda c: f"std_{c}")

    summ = pd.concat([appear, var_xyz, std_xyz], axis=1).fillna(0)
    summ["appear_ratio"] = summ["appear_frames"] / max(total_frames, 1)

    summ["pos_var_sum"] = summ["var_x"] + summ["var_y"] + summ["var_z"]
    summ["pos_std_sum"] = summ["std_x"] + summ["std_y"] + summ["std_z"]

    # 排序：先按出现率高，再按“动得多”
    summ = summ.sort_values(["appear_ratio", "pos_var_sum"], ascending=[False, False])
    summ.reset_index(inplace=True)  # point_id 变成列
    return summ


def auto_pick_points(summary: pd.DataFrame):
    """
    自动挑点策略：
    1) 固定点候选：出现率高 & pos_var_sum 小
    2) 运动点候选：出现率高 & pos_var_sum 大
    我们主要要运动点 top4。
    """
    # 出现率阈值：先取 >= 0.8（如果点很多且稳定，一般够用）
    cand = summary[summary["appear_ratio"] >= 0.80].copy()
    if len(cand) < 10:
        # 数据可能更稀疏，放宽
        cand = summary[summary["appear_ratio"] >= 0.50].copy()

    # 运动点：在候选里按 pos_var_sum 取前4
    moving = cand.sort_values("pos_var_sum", ascending=False).head(4)["point_id"].astype(int).tolist()

    # 固定点：在候选里按 pos_var_sum 取最小的前4（输出给你看）
    fixed = cand.sort_values("pos_var_sum", ascending=True).head(4)["point_id"].astype(int).tolist()

    return moving, fixed


def build_frame_table(df4: pd.DataFrame, target_ids):
    pos = df4.pivot_table(
        index="frame_idx",
        columns="point_id",
        values=["x", "y", "z"],
        aggfunc="first"
    )
    pos.columns = [f"{axis}_p{pid}" for axis, pid in pos.columns]
    pos = pos.sort_index()

    sensor = df4.groupby("frame_idx")[SENSOR_COLS].median().sort_index()

    frames = pos.join(sensor, how="inner").reset_index()

    cols = ["frame_idx"]
    for pid in target_ids:
        cols += [f"x_p{pid}", f"y_p{pid}", f"z_p{pid}"]
    cols += SENSOR_COLS
    cols = [c for c in cols if c in frames.columns]
    return frames[cols]


def drop_outlier_frames_by_diff(frames: pd.DataFrame, target_ids):
    frames = frames.sort_values("frame_idx").reset_index(drop=True)

    pos_cols = []
    for pid in target_ids:
        for axis in ["x", "y", "z"]:
            c = f"{axis}_p{pid}"
            if c in frames.columns:
                pos_cols.append(c)
    sensor_cols = [c for c in SENSOR_COLS if c in frames.columns]

    bad = np.zeros(len(frames), dtype=bool)

    if pos_cols:
        pos_mat = frames[pos_cols].to_numpy(dtype=float)
        dpos = np.diff(pos_mat, axis=0)
        dpos_norm = np.linalg.norm(dpos, axis=1)
        z = np.abs(robust_zscore_by_mad(dpos_norm))
        bad[1:] |= (z > POS_DIFF_MAD_Z)

    if sensor_cols:
        s_mat = frames[sensor_cols].to_numpy(dtype=float)
        ds = np.diff(s_mat, axis=0)
        ds_norm = np.linalg.norm(ds, axis=1)
        z = np.abs(robust_zscore_by_mad(ds_norm))
        bad[1:] |= (z > SENSOR_DIFF_MAD_Z)

    return frames.loc[~bad].reset_index(drop=True), bad


def main():
    # 1) 读入
    rows = []
    with open(FILE_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            r = parse_line(line)
            if r is not None:
                rows.append(r)
    if not rows:
        raise RuntimeError("没有解析到有效行。")

    df = pd.DataFrame(rows, columns=["point_id", "x", "y", "z", "v1", "v2", "v3", "v4"])

    # 2) 粗分帧（不依赖固定点）
    df = assign_frames_by_pid_cycle(df)

    # 3) 点统计（输出给你看）
    summary = summarize_points(df)
    summary.to_csv(OUT_SUMMARY_POINTS, index=False, encoding="utf-8-sig")
    print(f"[Info] 已输出点统计: {OUT_SUMMARY_POINTS}")

    # 4) 自动挑运动4点（并给出固定点候选供你核对）
    moving4, fixed4 = auto_pick_points(summary)
    print(f"[Auto] 运动点 top4: {moving4}")
    print(f"[Auto] 固定点候选 top4(最稳): {fixed4}")

    target_ids = TARGET_POINT_IDS if TARGET_POINT_IDS is not None else moving4
    target_ids = list(map(int, target_ids))
    print(f"[Use] 最终使用的4个目标点: {target_ids}")

    # 5) 只保留目标4点
    df4 = df[df["point_id"].isin(target_ids)].copy()

    # 6) 帧完整性：每帧必须4点齐全且唯一
    nunique = df4.groupby("frame_idx")["point_id"].nunique()
    good_frames = nunique[nunique == len(target_ids)].index
    df4 = df4[df4["frame_idx"].isin(good_frames)].copy()

    # 7) 传感器异常剔除（按帧聚合）
    sensor_by_frame = df4.groupby("frame_idx")[SENSOR_COLS].median()

    if DROP_IF_ALL_SENSOR_ZERO:
        all_zero_frames = sensor_by_frame.index[(sensor_by_frame.fillna(0).abs().sum(axis=1) == 0)]
        df4 = df4[~df4["frame_idx"].isin(all_zero_frames)].copy()

    if DROP_IF_V1_ZERO:
        v1_zero_frames = sensor_by_frame.index[np.isclose(sensor_by_frame["v1"].fillna(0).to_numpy(), 0.0)]
        df4 = df4[~df4["frame_idx"].isin(v1_zero_frames)].copy()

    # 8) 宽表 + 差分离群
    frames = build_frame_table(df4, target_ids)
    frames_clean, _ = drop_outlier_frames_by_diff(frames, target_ids)

    # 同步过滤长表
    good_set = set(frames_clean["frame_idx"].astype(int).tolist())
    df4_clean = df4[df4["frame_idx"].isin(good_set)].copy()

    # 9) 输出
    frames_clean.to_csv(OUT_FRAMES_CSV, index=False, encoding="utf-8-sig")
    df4_clean.to_csv(OUT_LONG_CSV, index=False, encoding="utf-8-sig")

    print("========== Done ==========")
    print(f"原始有效行数: {len(df)}")
    print(f"粗分帧总数: {df['frame_idx'].nunique()}")
    print(f"清洗后帧数: {len(frames_clean)}")
    print(f"输出: {OUT_FRAMES_CSV} / {OUT_LONG_CSV}")


if __name__ == "__main__":
    main()
