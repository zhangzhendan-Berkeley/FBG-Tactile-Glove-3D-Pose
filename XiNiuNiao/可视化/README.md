# Pose Visualization / 位姿可视化

## 中文说明

本目录用于对比动捕真值和模型预测的指尖位姿结果，支持逐帧播放、轨迹绘制、误差统计和论文图导出。

### 主要脚本

```text
visualize_all.py             skeleton and predicted pose comparison
visualize_gt_pred_gui.py     GUI for ground-truth/prediction comparison
visualize_pose24_gui.py      24D pose format viewer
可视化位姿_箭头.py            arrow-based pose visualization
可视化轨迹.py                 trajectory plotting
更好的轨迹.py                 refined trajectory figure generation
```

### 输入文件

常用输入包括：

```text
clean_glove_one_row_per_frame_cut.csv
processed_test_rot6d.txt
gt_tip_pose.txt
pred_tip_pose.txt
```

这些文件由预处理和推理脚本生成，不提交到 Git。

## English

This directory visualizes the ground-truth and predicted fingertip poses. It supports frame-by-frame playback, trajectory plotting, error statistics, and publication-quality figure export.

Prepare the exported preprocessing/inference files locally before running the scripts.
