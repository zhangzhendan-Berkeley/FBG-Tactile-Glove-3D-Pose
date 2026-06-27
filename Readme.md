# FBG-Tactile-Glove-3D-Pose

中文 | English

## 项目简介

本仓库整理了光纤弯曲传感手套的 Python 代码项目，用于中指末端六自由度位姿重建。代码覆盖数据采集预处理、刚体位姿求解、模型训练、推理可视化、鼠标点击微动作分析，以及微裂隙光波导仿真。

论文、LaTeX 模板、审稿回复、原始大体积数据和训练权重不包含在本仓库中。`data/`、`runs/`、模型权重和生成图像默认被 `.gitignore` 排除。复现实验时，请按各子目录 README 准备数据或重新生成中间文件。

## Project Overview

This repository contains the Python code for a fiber-optic bending-sensing glove for 6-DoF middle-fingertip pose reconstruction. It includes data preprocessing, rigid-body pose recovery, model training, inference visualization, mouse-click micro-motion analysis, and micro-slit optical-waveguide simulation.

Paper sources, LaTeX templates, review responses, large raw data, and trained checkpoints are not included. `data/`, `runs/`, checkpoints, and generated figures are ignored by default. Please prepare data or regenerate intermediate files according to each submodule README.

## 目录结构 / Repository Structure

```text
hgrc_fm_v3/                    Model training, baselines, inference
LuMoSDKPy/                     Data acquisition and preprocessing
XiNiuNiao/                     Visualization, analysis, plotting
simulation/                    Micro-slit/notch ray-tracing simulation
separated_calibration_figures/ Calibration plotting utilities
analyze_uncut_fiber_calibration.py
```

## 快速开始 / Quick Start

```bash
cd hgrc_fm_v3
pip install -r requirements.txt
python -m rigid_flow.train_mamba_coarse_only --config configs/rigid_config.yaml
python -m rigid_flow.train_mamba_with_flow --config configs/rigid_config.yaml --coarse_ckpt runs/mamba_coarse_only/best_model.pt
```

The commands above require prepared CSV files under `hgrc_fm_v3/data/`, which are intentionally not committed.

## 数据约定 / Data Convention

The model uses a temporal input window:

```text
[T, 12] = hand-back pose 9D + fiber channels 3D
target [9] = fingertip position 3D + rotation 6D
```

Earlier acquisition scripts may contain four-channel compatibility code because the acquisition board had four analog channels. The model-side cleaned pipeline uses the three active fiber channels corresponding to MCP, PIP, and DIP joints.

## 子模块 / Submodules

- `hgrc_fm_v3`: temporal pose reconstruction models, including Transformer, Mamba, Mamba + Flow Matching, and recent baseline models.
- `LuMoSDKPy`: acquisition and preprocessing scripts for motion-capture markers, fiber channels, rigid-body pose solving, and dataset splitting.
- `XiNiuNiao`: visualization tools for predicted/ground-truth pose, trajectory figures, mouse-click detection, and SOTA comparison plots.
- `simulation`: lightweight 2D ray-tracing simulation for different crack/notch geometries.
- `separated_calibration_figures`: standalone plotting scripts for calibration curves.

## License / 许可证

This code is released for academic and research use. Please cite the corresponding paper if this repository is useful for your work.
