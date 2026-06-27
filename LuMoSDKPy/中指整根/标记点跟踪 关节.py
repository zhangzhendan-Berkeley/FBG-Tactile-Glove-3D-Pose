# -*- coding: utf-8 -*-
"""
处理光纤形状传感手套动捕 CSV（扩展版：增加 2 个单点跟踪）
------------------------------------------------------------
输入格式（分号分隔）：
frame_idx; marker_id; x; y; z; ch1; ch2; ch3; ch4

输出：
1. clean_glove_one_row_per_frame.csv   -> 一帧一行
2. abnormal_frames_log.csv             -> 异常帧/异常点日志
3. id_change_log.csv                   -> ID变化日志

跟踪对象：
- 手背刚体 4 点
- 指尖刚体 4 点
- PIP 关节点 (近端指间关节)
- DIP 关节点 (远端指间关节)

作者建议：
- 先用这个版本跑通
- 新增两个关节点先按"单点连续跟踪"处理
- 如果后面你觉得还不够稳，再加几何约束/骨架约束
"""

import itertools
import numpy as np
import pandas as pd


# =========================================================
# 1. 配置区
# =========================================================

INPUT_CSV = r"sync_data_with_frame_mouse.csv"

OUT_CLEAN_CSV = r"clean_glove_one_row_per_frame.csv"
OUT_ABNORMAL_LOG = r"abnormal_frames_log.csv"
OUT_ID_CHANGE_LOG = r"id_change_log.csv"

# -------------------------
# 起始已知 ID（首帧）
# 左上开始顺时针
# -------------------------
START_BACK_IDS = [100015, 106558, 106789, 100017]   # back_lt, back_rt, back_rb, back_lb
START_TIP_IDS  = [105973, 106674, 106749, 105974]   # tip_lt,  tip_rt,  tip_rb,  tip_lb

# 新增两个单点：改为 PIP 和 DIP
START_JOINT_IDS = {
    "pip":  100019,   # PIP关节 (近端指间关节)
    "dip":  105972,   # DIP关节 (远端指间关节)
}

# -------------------------
# 结束已知 ID（末帧人工核对用）
# -------------------------
END_BACK_IDS = [105973, 106972, 106970, 105974]
END_TIP_IDS  = [100015, 106888, 106789, 100017]

END_JOINT_IDS = {
    "pip":  100019,   # PIP关节末帧ID（如已知）
    "dip":  105972,   # DIP关节末帧ID（如已知）
}

BACK_NAMES = ["back_lt", "back_rt", "back_rb", "back_lb"]
TIP_NAMES  = ["tip_lt", "tip_rt", "tip_rb", "tip_lb"]
JOINT_NAMES = ["pip", "dip"]  # 改为 PIP 和 DIP

ALL_NAMES = BACK_NAMES + TIP_NAMES + JOINT_NAMES

# -------------------------
# 阈值参数（mm）
# -------------------------

# 手背/指尖刚体点瞬移阈值
JUMP_THRESH_MM = 50.0

# 新增单点的瞬移阈值（一般可稍微放宽）
JUMP_THRESH_JOINT_MM = 60.0

# 重新捕获半径
REACQUIRE_THRESH_BACK = 20.0
REACQUIRE_THRESH_TIP = 35.0
REACQUIRE_THRESH_JOINT = 45.0

# 群中心约束半径
GROUP_CENTER_RADIUS_BACK = 80.0
GROUP_CENTER_RADIUS_TIP = 120.0

# 单点候选搜索半径（先小范围找）
JOINT_SEARCH_RADIUS = 80.0

# 若本帧某点异常/缺失，是否先用上一帧临时填充
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
    return np.array([
        g["ch1"].median(),
        g["ch2"].median(),
        g["ch3"].median(),
        g["ch4"].median(),
    ], dtype=float)


# =========================================================
# 3. 刚体四点匹配逻辑
# =========================================================

def build_candidates(g, used_ids, group_center_prev=None, group_radius=None):
    """
    从当前帧中筛一批候选点
    """
    candidates = []
    for _, r in g.iterrows():
        pid = int(r["marker_id"])
        if pid in used_ids:
            continue

        xyz = get_xyz(r)

        if group_center_prev is not None and group_radius is not None:
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
    对 4 个点做最优匹配
    代价：到上一帧对应点的距离
    若距离 > jump_thresh，则视为不合法
    """
    assigned = {name: None for name in track_names}

    valid_tracks = [n for n in track_names if prev_positions.get(n, None) is not None]
    if len(valid_tracks) == 0 or len(candidates) == 0:
        return assigned

    best_cost = float("inf")
    best_assign = None

    k = min(len(candidates), len(track_names))

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
# 4. 单点跟踪逻辑（PIP 和 DIP）
# =========================================================

def track_single_point(
    g,
    used_ids,
    prev_xyz,
    search_radius,
    jump_thresh,
    reacquire_thresh,
    anchor_center=None,
    anchor_radius=None
):
    """
    跟踪单个逻辑点：
    1) 先在 prev_xyz 附近 search_radius 内找最近点
    2) 如果找不到，再在更大半径 reacquire_thresh 内找
    3) 可选 anchor_center / anchor_radius 进一步限制候选点范围
    """
    if prev_xyz is None:
        return None

    # 第一轮：小范围搜索
    best = None
    best_d = float("inf")

    for _, r in g.iterrows():
        pid = int(r["marker_id"])
        if pid in used_ids:
            continue

        xyz = get_xyz(r)

        if anchor_center is not None and anchor_radius is not None:
            if dist(xyz, anchor_center) > anchor_radius:
                continue

        d = dist(xyz, prev_xyz)
        if d < best_d and d <= search_radius:
            best_d = d
            best = {
                "id": pid,
                "xyz": xyz,
                "row": r
            }

    if best is not None:
        return best

    # 第二轮：扩大找回
    best = None
    best_d = float("inf")

    for _, r in g.iterrows():
        pid = int(r["marker_id"])
        if pid in used_ids:
            continue

        xyz = get_xyz(r)

        if anchor_center is not None and anchor_radius is not None:
            if dist(xyz, anchor_center) > anchor_radius:
                continue

        d = dist(xyz, prev_xyz)
        if d < best_d and d <= reacquire_thresh:
            best_d = d
            best = {
                "id": pid,
                "xyz": xyz,
                "row": r
            }

    if best is not None and dist(best["xyz"], prev_xyz) <= jump_thresh:
        return best

    return best


# =========================================================
# 5. 主流程
# =========================================================

def process():
    df = load_csv(INPUT_CSV)
    frame_list = sorted(df["frame_idx"].unique().tolist())
    print(f"总帧数: {len(frame_list)}")

    grouped = {fi: g.copy() for fi, g in df.groupby("frame_idx")}

    # ========= 初始化：首帧 =========
    first_frame = grouped[frame_list[0]]
    first_id_map = rows_to_id_map(first_frame)

    track_state = {}
    prev_positions = {}
    prev_ids = {}

    # 手背 + 指尖
    start_ids_main = START_BACK_IDS + START_TIP_IDS
    main_names = BACK_NAMES + TIP_NAMES

    for name, pid in zip(main_names, start_ids_main):
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

    # 新增两个关节点（PIP 和 DIP）
    for name in JOINT_NAMES:
        pid = START_JOINT_IDS.get(name, -1)
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
            print(f"[警告] 首帧中未找到新增点 {name} id={pid}")

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

        # =====================================================
        # A. 手背 4 点
        # =====================================================
        back_center_prev = group_center(prev_positions, BACK_NAMES)
        back_candidates = build_candidates(
            g, used_ids,
            group_center_prev=back_center_prev,
            group_radius=GROUP_CENTER_RADIUS_BACK
        )

        back_assigned = assign_points_to_tracks(
            back_candidates, BACK_NAMES, prev_positions, JUMP_THRESH_MM
        )
        used_ids |= set(v["id"] for v in back_assigned.values() if v is not None)

        back_assigned = reacquire_missing(
            g, used_ids, back_assigned, BACK_NAMES, prev_positions, REACQUIRE_THRESH_BACK
        )
        used_ids |= set(v["id"] for v in back_assigned.values() if v is not None)

        # =====================================================
        # B. 指尖 4 点
        # =====================================================
        tip_center_prev = group_center(prev_positions, TIP_NAMES)
        tip_candidates = build_candidates(
            g, used_ids,
            group_center_prev=tip_center_prev,
            group_radius=GROUP_CENTER_RADIUS_TIP
        )

        tip_assigned = assign_points_to_tracks(
            tip_candidates, TIP_NAMES, prev_positions, JUMP_THRESH_MM
        )
        used_ids |= set(v["id"] for v in tip_assigned.values() if v is not None)

        tip_assigned = reacquire_missing(
            g, used_ids, tip_assigned, TIP_NAMES, prev_positions, REACQUIRE_THRESH_TIP
        )
        used_ids |= set(v["id"] for v in tip_assigned.values() if v is not None)

        # =====================================================
        # C. 两个新增单点（PIP 和 DIP）
        # =====================================================
        joint_assigned = {}

        # 这里可选地把"指尖组中心"作为附加先验
        finger_anchor = group_center(prev_positions, TIP_NAMES)

        for name in JOINT_NAMES:
            prev_xyz = prev_positions.get(name, None)

            cur = track_single_point(
                g=g,
                used_ids=used_ids,
                prev_xyz=prev_xyz,
                search_radius=JOINT_SEARCH_RADIUS,
                jump_thresh=JUMP_THRESH_JOINT_MM,
                reacquire_thresh=REACQUIRE_THRESH_JOINT,
                anchor_center=finger_anchor,     # 可作为空间先验
                anchor_radius=180.0 if finger_anchor is not None else None
            )

            joint_assigned[name] = cur
            if cur is not None:
                used_ids.add(cur["id"])

        # 汇总
        cur_assigned = {}
        cur_assigned.update(back_assigned)
        cur_assigned.update(tip_assigned)
        cur_assigned.update(joint_assigned)

        # 统计本帧找到多少有用点
        expected_n = len(ALL_NAMES)
        n_found = sum(1 for v in cur_assigned.values() if v is not None)
        if n_found < expected_n:
            abnormal_logs.append({
                "frame_idx": fi,
                "type": "insufficient_useful_points",
                "detail": f"only_found_{n_found}_of_{expected_n}"
            })

        # =====================================================
        # D. 写出 + 异常检查
        # =====================================================
        for name in ALL_NAMES:
            cur = cur_assigned.get(name, None)
            prev_xyz = prev_positions.get(name, None)
            prev_id = prev_ids.get(name, None)

            if name in JOINT_NAMES:
                jump_thresh_this = JUMP_THRESH_JOINT_MM
            else:
                jump_thresh_this = JUMP_THRESH_MM

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
                    if d > jump_thresh_this:
                        bad_flag = True
                        bad_reason = f"jump_gt_{jump_thresh_this}mm:{d:.3f}"

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

        # =====================================================
        # E. 更新 prev
        # =====================================================
        for name in ALL_NAMES:
            x = row_out[f"{name}_x"]
            y = row_out[f"{name}_y"]
            z = row_out[f"{name}_z"]
            pid = row_out[f"{name}_id"]

            if pd.isna(x) or pd.isna(y) or pd.isna(z):
                pass
            else:
                prev_positions[name] = np.array([x, y, z], dtype=float)
                prev_ids[name] = None if pid == -1 else int(pid)

        clean_rows.append(row_out)

    # =====================================================
    # 6. 汇总输出
    # =====================================================
    df_clean = pd.DataFrame(clean_rows)
    df_abn = pd.DataFrame(abnormal_logs)
    df_idchg = pd.DataFrame(id_change_logs)

    # 可选：线性插值
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
    for n in JOINT_NAMES:
        print(f"{n}: {END_JOINT_IDS.get(n, -1)}")


if __name__ == "__main__":
    process()