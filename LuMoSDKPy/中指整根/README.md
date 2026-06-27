# Full Middle-Finger Pipeline / 中指整根数据流程

## 中文说明

本目录用于完整中指实验的数据处理，包含同步采集、标记点跟踪、刚体中心与姿态计算、可视化检查、数据集划分和统计量计算。

### 主要脚本

```text
采样v2.py                         synchronized acquisition
标记点跟踪.py                     marker identity tracking
标记点跟踪 关节.py                marker tracking with optional joint markers
帧裁剪.py                         trim valid frame ranges
计算刚体中心位置与四元数姿态.py   rigid-body center and quaternion recovery
split_dataset.py                  train/val/test split
compute_stats.py                  dataset statistics
geometry.py                       pose and rotation utilities
数据集可视化.py                   marker/rigid-body visualization
数据集可视化 带关节.py            visualization with PIP/DIP markers
可视化刚体中心点.py               rigid-body pose viewer
商业手套标定.py                   optional commercial glove calibration
```

### 输入与输出

原始采集文件通常包含每帧的 marker 坐标和传感器电压。处理后的文件会被转换为模型训练使用的 CSV：

```text
hand-back pose 9D + fingertip pose 9D + fiber channels
```

数据文件默认被 `.gitignore` 排除，避免提交大体积或未脱敏实验数据。

## English

This directory implements the full middle-finger preprocessing pipeline, including synchronized acquisition, marker identity tracking, rigid-body pose recovery, visualization checks, dataset splitting, and statistics computation.

The final training CSV stores hand-back pose, fingertip pose, and fiber-sensor channels. Raw and processed data files are intentionally ignored by Git.
