# Mouse-Click Micro-Motion Analysis / 鼠标点击微动作分析

## 中文说明

本目录用于基于模型预测的指尖竖直位移检测鼠标点击动作。脚本支持点击曲线播放、峰值检测、事件查看和调试统计。

### 主要脚本

```text
detect_clicks_from_mocap.py      click detection from reconstructed motion
viewer_with_click_events.py      viewer with detected click events
曲线分析.py                      signal analysis
曲线播放 进度条.py               playback with progress bar
点击曲线播放.py                  click-signal playback
```

### 基本思路

1. 从预测指尖位置中提取竖直方向位移。
2. 去趋势并用稳健尺度归一化。
3. 对向下点击响应做整流和平滑。
4. 使用阈值、峰值和下降沿规则检测点击事件。

## English

This directory detects mouse-click micro-motions from reconstructed fingertip displacement. The pipeline detrends and normalizes the vertical fingertip trajectory, rectifies the click response, smooths it, and detects click events with threshold/peak/falling-edge rules.
