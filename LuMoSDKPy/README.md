# Data Acquisition and Preprocessing / 数据采集与预处理

## 中文说明

本目录包含实验数据采集和预处理脚本，用于将光学动捕标记点、光纤传感器通道和可选商业手套数据整理为模型训练所需的数据格式。

## English

This directory contains data acquisition and preprocessing scripts. The scripts convert optical motion-capture markers, fiber-sensor channels, and optional commercial-glove signals into the format required by the training pipeline.

## 子目录 / Subdirectories

- `中指整根/`: full middle-finger acquisition, marker tracking, rigid-body pose solving, and dataset splitting.
- `单关节/`: single-joint sensor calibration, angle checking, hysteresis analysis, and manual data pruning.

## 典型流程 / Typical Pipeline

```text
acquisition -> marker tracking -> frame trimming -> rigid-body pose solving -> format conversion -> train/val/test split
采集 -> 标记点跟踪 -> 帧裁剪 -> 刚体位姿求解 -> 格式转换 -> 数据集划分
```

Raw synchronized CSV/TXT files are ignored by Git. Please place your local data under the paths expected by each script.

原始同步数据文件不随仓库提交。请根据脚本中的默认路径或命令行参数放置本地数据。
