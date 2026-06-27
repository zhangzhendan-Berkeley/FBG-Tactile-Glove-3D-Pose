# Micro-Slit Optical Simulation / 微裂隙光学仿真

## 中文说明

本目录包含一个轻量级 2D 几何光线追迹仿真，用于比较不同裂隙或缺口形状对弯曲角-光损失曲线的影响。仿真对象为 TPU 光波导截面展开后的二维场景，包含闭合微裂隙、矩形去除缺口和 V 形去除缺口。

### 运行

```bash
python crack_shape_raytrace.py
```

### 输出

脚本会在 `crack_shape_results/` 中生成示意图、损失曲线、CSV 和 summary。生成图像与 CSV 默认不提交到 Git。

### 解释

闭合微裂隙在伸直状态下保持较完整的全反射路径，初始光损失较小；弯曲后裂隙张开导致光泄露增加。材料去除型矩形或 V 形缺口在未弯曲时已经破坏部分光路，因此初始损失更高、可用动态范围更小。

## English

This directory provides a lightweight 2D geometric ray-tracing simulation for comparing how different crack/notch geometries affect the bending-angle versus optical-loss response. The simulated TPU waveguide includes closed micro-slits, rectangular removed notches, and V-shaped removed notches.

Run:

```bash
python crack_shape_raytrace.py
```

The script writes figures, CSV files, and a summary under `crack_shape_results/`. Generated outputs are ignored by Git.
