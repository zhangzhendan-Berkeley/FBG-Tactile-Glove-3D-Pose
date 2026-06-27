# Calibration Figure Utilities / 标定曲线绘图工具

## 中文说明

本目录用于将传感器标定数据绘制为独立图像，便于检查归一化前后的电压-角度曲线、误差带和拟合曲线。该目录主要用于图形调试，不依赖论文 LaTeX 编译。

### 运行

```bash
python plot_separated_calibration_curves.py
```

脚本会读取本地 CSV 并生成 PDF/PNG 图像。CSV 和图像默认不提交到 Git。

## English

This directory contains standalone utilities for plotting calibration curves, error bands, and fitted voltage-angle responses. It is intended for figure debugging and does not depend on the LaTeX manuscript build.

Run:

```bash
python plot_separated_calibration_curves.py
```

Local CSV files and generated figures are ignored by Git.
