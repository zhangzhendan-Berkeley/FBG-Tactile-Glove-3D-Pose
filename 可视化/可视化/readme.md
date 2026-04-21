# 手指位姿重建可视化工具

## 📌 项目简介

本项目提供两个用于手指位姿重建结果分析的可视化工具：

* 骨架 + 位姿对比可视化
* 轨迹对比可视化

适用于光纤手套 + 动捕标定数据分析与论文作图。

---

## 📦 文件说明

| 文件                      | 功能              |
| ----------------------- | --------------- |
| visualize_all.py        | 骨架 + 位姿对比（逐帧）   |
| visualize_trajectory.py | GT vs Pred 轨迹对比 |

---

## ⚙️ 环境依赖

```
pip install numpy pandas matplotlib
```

---

## 📂 数据格式

### 1️⃣ 骨架数据（CSV）

包含字段：

```
frame_idx
back_lt_x, back_lt_y, back_lt_z
...
tip_lt_x, ...
pip_x, pip_y, pip_z
dip_x, dip_y, dip_z
```

⚠️ 坐标为 **yzx 顺序（程序自动转换）**

---

### 2️⃣ 位姿数据（TXT）

#### test_rot6d.txt

```
frame_idx, back_pos(3), back_rot6d(6), ...
```

#### gt_tip_pose.txt / pred_tip_pose.txt

```
frame_idx, pos(3), rot6d(6)
```

---

## 🚀 使用方法

### ▶️ 骨架 + 位姿对比

```
python visualize_all.py
```

默认路径：

```
clean_glove_one_row_per_frame_cut.csv
processed_test_rot6d.txt
gt_tip_pose.txt
pred_tip_pose.txt
```

---

### ▶️ 轨迹对比

```
python visualize_trajectory.py
```

---

## 🎮 功能说明

### visualize_all.py

* 左图：骨架（Back / PIP / DIP / Tip）
* 右图：预测位姿（位置 + 方向）
* 显示误差：

  * Position Error（mm）
  * Rotation Error（deg）

支持：

* 播放 / 暂停
* 单帧切换
* 进度条拖动
* 倍速播放
* 视角同步
* 缩放
* 导出 EPS / PDF

---

### visualize_trajectory.py

* GT / Pred 轨迹对比（3D）
* 颜色表示时间顺序
* 起点 / 终点标记

支持：

* 时间区间选择
* 坐标轴调整
* 图例控制
* 高质量论文图导出

输出统计：

* Mean / Max / Min / Std Error
* 轨迹长度（GT / Pred）

---

## 🖱️ 操作说明

### 通用

* 鼠标滚轮：缩放
* 鼠标拖动：旋转

### 快捷键（轨迹工具）

* Ctrl + E：保存 EPS
* Ctrl + P：保存 PDF
* Ctrl + A：自动坐标
* Ctrl + Q：退出

---

## 📌 注意事项

* 输入坐标需为 **yzx**
* 旋转为 **rot6D 表示**
* 自动进行帧对齐
* 单位建议使用 mm

---

## 📊 推荐用途

| 场景     | 工具                   |
| ------ | -------------------- |
| 模型效果展示 | visualize_all        |
| 论文图绘制  | visualize_trajectory |

---

## 📄 License

仅用于科研用途。
