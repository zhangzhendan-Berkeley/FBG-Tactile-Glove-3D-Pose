# -*- coding: utf-8 -*-
"""
处理光纤形状传感手套动捕 CSV

输入格式（分号分隔）：
frame_idx; marker_id; x; y; z; ch1; ch2; ch3; ch4

输出：
1. clean_glove_one_row_per_frame.csv   -> 一帧一行
2. abnormal_frames_log.csv             -> 异常帧/异常点日志
3. id_change_log.csv                   -> ID变化日志

作者建议：
- 先跑这个版本
- 如果结果基本正确，再根据你的数据特点继续加“正方形几何约束”
"""

import itertools
import numpy as np
import pandas as pd


# =========================================================
# 1. 配置区
# =========================================================

INPUT_CSV = r"sync_data_with_frame_more_points.csv"

OUT_CLEAN_CSV = r"clean_glove_one_row_per_frame.csv"
OUT_ABNORMAL_LOG = r"abnormal_frames_log.csv"
OUT_ID_CHANGE_LOG = r"id_change_log.csv"

# 起始已知 ID（左上开始顺时针）
START_BACK_IDS = [103299, 103298, 100758, 103349]   # back_lt, back_rt, back_rb, back_lb
START_TIP_IDS  = [103390, 103388, 103389, 103391]   # tip_lt,  tip_rt,  tip_rb,  tip_lb

# 结束已知 ID（可用于最后核对）
END_BACK_IDS = [103067, 103069, 103065, 103096]
END_TIP_IDS  = [103031, 103090, 103058, 103088]

BACK_NAMES = ["back_lt", "back_rt", "back_rb", "back_lb"]
TIP_NAMES  = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]
ALL_NAMES  = BACK_NAMES + TIP_NAMES

# 瞬移阈值（mm）
JUMP_THRESH_MM = 50.0

# 重新捕获时允许的更大搜索半径（mm）
REACQUIRE_THRESH_BACK = 20.0
REACQUIRE_THRESH_TIP = 35.0

# 群中心约束，过滤掉离该组整体太远的候选点
GROUP_CENTER_RADIUS_BACK = 80.0
GROUP_CENTER_RADIUS_TIP = 120.0

# 若本帧某点异常/缺失，是否先用上一帧临时填充
# 建议先 False，保留 NaN，后面统一插值
FILL_BAD_WITH_PREV = False

# 最后是否做插值
DO_INTERPOLATE = True

# 最大允许插值连续缺失长度（帧）
MAX_INTERP_GAP = 10


# =========================================================
# 2. 基础工具函数
# =========================================================

def load_csv(path):
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # 转数字
    num_cols = ["frame_idx", "marker_id", "x", "y", "z", "ch1", "ch2", "ch3", "ch4"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["frame_idx", "marker_id", "x", "y", "z"]).copy()
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["marker_id"] = df["marker_id"].astype(int)
    return df


def dist(a, b):
    return float(np.linalg.norm(a - b))


def get_xyz(row):
    return np.array([row["x"], row["y"], row["z"]], dtype=float)


def group_center(prev_positions, names):
    pts = []
    for n in names:
        p = prev_positions.get(n, None)
        if p is not None and not np.isnan(p).any():
            pts.append(p)
    if len(pts) == 0:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def rows_to_id_map(g):
    mp = {}
    for _, r in g.iterrows():
        pid = int(r["marker_id"])
        if pid not in mp:
            mp[pid] = r
    return mp


def get_frame_voltage(g):
    # 同一帧每行电压一般一样，取中位数更稳
    return np.array([
        g["ch1"].median(),
        g["ch2"].median(),
        g["ch3"].median(),
        g["ch4"].median(),
    ], dtype=float)


# =========================================================
# 3. 匹配逻辑
# =========================================================

def build_candidates(g, used_ids, group_center_prev, group_radius):
    """
    从当前帧中筛一批候选点
    """
    candidates = []
    for _, r in g.iterrows():
        pid = int(r["marker_id"])
        if pid in used_ids:
            continue
        xyz = get_xyz(r)

        if group_center_prev is not None:
            if dist(xyz, group_center_prev) > group_radius:
                continue

        candidates.append({
            "id": pid,
            "xyz": xyz,
            "row": r
        })
    return candidates


def assign_points_to_tracks(candidates, track_names, prev_positions, jump_thresh):
    """
    在小规模情况下对 4 个点做最优匹配
    代价：到上一帧对应点的距离
    若距离 > jump_thresh，则视为不合法
    """
    assigned = {name: None for name in track_names}

    valid_tracks = [n for n in track_names if prev_positions.get(n, None) is not None]
    if len(valid_tracks) == 0 or len(candidates) == 0:
        return assigned

    # 只尝试候选数不太大时穷举；你每帧点数应该不至于太多
    # 如果候选很多，可进一步先按最近邻预筛
    best_cost = float("inf")
    best_assign = None

    k = min(len(candidates), len(track_names))

    # 选 k 个候选去匹配 k 个轨迹
    for cand_subset in itertools.combinations(range(len(candidates)), k):
        subset = [candidates[i] for i in cand_subset]

        for perm in itertools.permutations(range(k), k):
            cur = {name: None for name in track_names}
            total_cost = 0.0
            feasible = True

            for i in range(k):
                name = track_names[i]
                c = subset[perm[i]]
                prev = prev_positions.get(name, None)

                if prev is None:
                    feasible = False
                    break

                d = dist(c["xyz"], prev)
                if d > jump_thresh:
                    feasible = False
                    break

                total_cost += d
                cur[name] = c

            if feasible and total_cost < best_cost:
                best_cost = total_cost
                best_assign = cur

    if best_assign is not None:
        return best_assign

    return assigned


def reacquire_missing(g, used_ids, assigned, track_names, prev_positions, reacquire_thresh):
    """
    对没匹配上的点，再用更大阈值尝试找回
    """
    for name in track_names:
        if assigned[name] is not None:
            continue

        prev = prev_positions.get(name, None)
        if prev is None:
            continue

        best = None
        best_d = float("inf")

        for _, r in g.iterrows():
            pid = int(r["marker_id"])
            if pid in used_ids:
                continue

            xyz = get_xyz(r)
            d = dist(xyz, prev)
            if d < best_d and d <= reacquire_thresh:
                best_d = d
                best = {
                    "id": pid,
                    "xyz": xyz,
                    "row": r
                }

        if best is not None:
            assigned[name] = best
            used_ids.add(best["id"])

    return assigned


# =========================================================
# 4. 主流程
# =========================================================

def process():
    df = load_csv(INPUT_CSV)
    frame_list = sorted(df["frame_idx"].unique().tolist())
    print(f"总帧数: {len(frame_list)}")

    # 每帧分组
    grouped = {fi: g.copy() for fi, g in df.groupby("frame_idx")}

    # 初始化：从第一帧找起始 ID
    first_frame = grouped[frame_list[0]]
    first_id_map = rows_to_id_map(first_frame)

    start_ids = START_BACK_IDS + START_TIP_IDS
    track_state = {}
    prev_positions = {}
    prev_ids = {}

    for name, pid in zip(ALL_NAMES, start_ids):
        if pid in first_id_map:
            r = first_id_map[pid]
            xyz = get_xyz(r)
            track_state[name] = {"id": pid, "xyz": xyz}
            prev_positions[name] = xyz
            prev_ids[name] = pid
        else:
            track_state[name] = {"id": None, "xyz": None}
            prev_positions[name] = None
            prev_ids[name] = None
            print(f"[警告] 首帧中未找到起始点 {name} id={pid}")

    clean_rows = []
    abnormal_logs = []
    id_change_logs = []

    for fi in frame_list:
        g = grouped[fi]
        row_out = {"frame_idx": fi}
        volt = get_frame_voltage(g)
        row_out["ch1"] = float(volt[0])
        row_out["ch2"] = float(volt[1])
        row_out["ch3"] = float(volt[2])
        row_out["ch4"] = float(volt[3])

        used_ids = set()

        # ========= 手背 =========
        back_center_prev = group_center(prev_positions, BACK_NAMES)
        back_candidates = build_candidates(
            g, used_ids, back_center_prev, GROUP_CENTER_RADIUS_BACK
        )
        back_assigned = assign_points_to_tracks(
            back_candidates, BACK_NAMES, prev_positions, JUMP_THRESH_MM
        )
        used_ids |= set(v["id"] for v in back_assigned.values() if v is not None)

        back_assigned = reacquire_missing(
            g, used_ids, back_assigned, BACK_NAMES, prev_positions, REACQUIRE_THRESH_BACK
        )
        used_ids |= set(v["id"] for v in back_assigned.values() if v is not None)

        # ========= 指尖 =========
        tip_center_prev = group_center(prev_positions, TIP_NAMES)
        tip_candidates = build_candidates(
            g, used_ids, tip_center_prev, GROUP_CENTER_RADIUS_TIP
        )
        tip_assigned = assign_points_to_tracks(
            tip_candidates, TIP_NAMES, prev_positions, JUMP_THRESH_MM
        )
        used_ids |= set(v["id"] for v in tip_assigned.values() if v is not None)

        tip_assigned = reacquire_missing(
            g, used_ids, tip_assigned, TIP_NAMES, prev_positions, REACQUIRE_THRESH_TIP
        )
        used_ids |= set(v["id"] for v in tip_assigned.values() if v is not None)

        cur_assigned = {}
        cur_assigned.update(back_assigned)
        cur_assigned.update(tip_assigned)

        # 统计这一帧找到多少有用点
        n_found = sum(1 for v in cur_assigned.values() if v is not None)
        if n_found < 8:
            abnormal_logs.append({
                "frame_idx": fi,
                "type": "insufficient_useful_points",
                "detail": f"only_found_{n_found}_of_8"
            })

        # ========= 写出 + 异常检查 =========
        for name in ALL_NAMES:
            cur = cur_assigned.get(name, None)
            prev_xyz = prev_positions.get(name, None)
            prev_id = prev_ids.get(name, None)

            bad_flag = False
            bad_reason = None

            if cur is None:
                bad_flag = True
                bad_reason = "missing_point"
            else:
                cur_xyz = cur["xyz"]
                cur_id = cur["id"]

                if prev_xyz is not None:
                    d = dist(cur_xyz, prev_xyz)
                    if d > JUMP_THRESH_MM:
                        bad_flag = True
                        bad_reason = f"jump_gt_{JUMP_THRESH_MM}mm:{d:.3f}"

                # 记录 ID 变化
                if (prev_id is not None) and (cur_id != prev_id):
                    id_change_logs.append({
                        "frame_idx": fi,
                        "logical_point": name,
                        "old_id": prev_id,
                        "new_id": cur_id
                    })

            if bad_flag:
                abnormal_logs.append({
                    "frame_idx": fi,
                    "type": "bad_point",
                    "detail": f"{name}:{bad_reason}"
                })

                if FILL_BAD_WITH_PREV and prev_xyz is not None:
                    out_xyz = prev_xyz.copy()
                    out_id = prev_id if prev_id is not None else -1
                else:
                    out_xyz = np.array([np.nan, np.nan, np.nan], dtype=float)
                    out_id = -1
            else:
                out_xyz = cur["xyz"]
                out_id = cur["id"]

            row_out[f"{name}_id"] = int(out_id) if out_id != -1 else -1
            row_out[f"{name}_x"] = float(out_xyz[0]) if not np.isnan(out_xyz[0]) else np.nan
            row_out[f"{name}_y"] = float(out_xyz[1]) if not np.isnan(out_xyz[1]) else np.nan
            row_out[f"{name}_z"] = float(out_xyz[2]) if not np.isnan(out_xyz[2]) else np.nan

        # ========= 这一帧写完后，更新 prev =========
        for name in ALL_NAMES:
            x = row_out[f"{name}_x"]
            y = row_out[f"{name}_y"]
            z = row_out[f"{name}_z"]
            pid = row_out[f"{name}_id"]

            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                # 缺失就不更新 prev，保留上一有效帧位置
                pass
            else:
                prev_positions[name] = np.array([x, y, z], dtype=float)
                prev_ids[name] = None if pid == -1 else int(pid)

        clean_rows.append(row_out)

    df_clean = pd.DataFrame(clean_rows)
    df_abn = pd.DataFrame(abnormal_logs)
    df_idchg = pd.DataFrame(id_change_logs)

    # ========= 可选：线性插值 =========
    if DO_INTERPOLATE:
        coord_cols = []
        for name in ALL_NAMES:
            coord_cols += [f"{name}_x", f"{name}_y", f"{name}_z"]

        for col in coord_cols:
            df_clean[col] = df_clean[col].interpolate(
                method="linear",
                limit=MAX_INTERP_GAP,
                limit_direction="both"
            )

    # 保存
    df_clean.to_csv(OUT_CLEAN_CSV, index=False, encoding="utf-8-sig")
    df_abn.to_csv(OUT_ABNORMAL_LOG, index=False, encoding="utf-8-sig")
    df_idchg.to_csv(OUT_ID_CHANGE_LOG, index=False, encoding="utf-8-sig")

    print(f"已保存: {OUT_CLEAN_CSV}")
    print(f"已保存: {OUT_ABNORMAL_LOG}")
    print(f"已保存: {OUT_ID_CHANGE_LOG}")

    # 末帧核对
    last = df_clean.iloc[-1]
    print("\n末帧识别结果：")
    for name in ALL_NAMES:
        print(f"{name}: {last[f'{name}_id']}")

    print("\n期望结束ID（供人工核对）：")
    for n, i in zip(BACK_NAMES, END_BACK_IDS):
        print(f"{n}: {i}")
    for n, i in zip(TIP_NAMES, END_TIP_IDS):
        print(f"{n}: {i}")


if __name__ == "__main__":
    process()