# Single-Joint Calibration / 单关节标定

## 中文说明

本目录用于单个光纤弯曲传感单元的标定、迟滞、重复性和数据清洗分析。它主要服务于传感器层面的电压-角度曲线、loading/unloading 曲线和异常点剔除。

### 主要脚本

```text
采集板.py                                acquisition-board test
test_LuMo_Arduino.py                    serial communication test
calculate angle.py                      angle calculation
check_angle.py                          angle sanity check
check_cut.py                            segment/cut inspection
data_cut.py                             data trimming
manual_prune_viewer.py                  manual outlier pruning GUI
prune_scatter_v_angle.py                scatter-based pruning
final_outlier_removal_by_angle_error.py final outlier removal
analyze_sensor_calibration.py           calibration metrics and plots
calibrate_v1_angle_hysteresis.py        loading/unloading hysteresis analysis
check_hysteresis_v1_angle.py            hysteresis validation
convert_to_rigid_mamba_txt.py           convert calibration data for model tools
Transformer.py                          legacy single-joint baseline
visualize_timeseries_slider.py          interactive time-series viewer
```

### 输出

脚本会生成标定曲线、迟滞曲线和清洗后的中间 CSV。生成结果默认不提交到 Git。

## English

This directory contains single-joint calibration and cleaning utilities for a fiber bending-sensing unit. It is mainly used for voltage-angle calibration, loading/unloading hysteresis analysis, repeatability checks, and outlier removal.

Generated calibration CSV files and figures are ignored by Git.
