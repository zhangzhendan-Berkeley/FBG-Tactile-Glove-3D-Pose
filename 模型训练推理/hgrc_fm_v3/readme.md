# 手指末端位姿重建模型训练说明（Training README）

---

## 📌 项目概述

本模块用于训练基于光纤传感数据的手指末端位姿重建模型。

模型输入为时间序列数据：

```text
back_seq: [T,13] = 手背位置(3) + 手背旋转6D(6) + 传感器(4)
```

模型输出为：

```text
y9: [9] = 指尖位置(3) + 指尖旋转6D(6)
```

数据由 `RigidSeqDataset` 自动构建滑动窗口样本 

---

## 配置
- `configs/rigid_schema.yaml`：列索引（无表头）；
- `configs/rigid_config.yaml`：
  - 选择训练/测试文件（支持多文件），每个文件可设置 `subject_id`；
  - `window_size` / `window_stride`；
  - `sensor_scale`（将原始传感器值除以该值再 clamp 到 [0,1]）；
  - 训练超参 & 损失权重。

---

## 🧠 模型结构

核心模型定义在：

```text
models.py
```

支持多种时序编码器：

* GRU
* TCN
* Transformer
* Mamba（推荐）

👉 统一模型结构：

```text
Encoder → Coarse Head → (Flow Residual 可选)
```

说明：

* Encoder：提取时序特征
* Coarse Head：预测基础位姿
* Flow：学习残差分布
---

## 📂 代码结构

```text
rigid_flow/
│
├── data.py                  # 数据集构建（滑窗 + 坐标转换）
├── geometry.py              # 坐标变换 / rot6D / 四元数
├── models.py                # 模型结构（Mamba + Flow）
│
├── train_transformer_with_data_py.py   # Transformer baseline
├── train_mamba_coarse_only.py          # Mamba 粗预测
├── train_mamba_with_flow.py            # Mamba + Flow
│
└── infer_mamba_with_flow_csv.py        # 推理脚本
```

---

## ⚙️ 训练流程总览

```text
1. 数据加载（滑窗）
2. 标准化（SeqStandardizer）
3. 模型训练（coarse / flow）
4. 验证评估（位置 + 旋转误差）
5. 保存 best_model.pt
```

---

## 📊 数据处理机制

### 输入构建

每帧：

```text
[back_pos(3), back_rot6D(6), sensor(4)] → 13维
```

滑窗：

```text
[T,13] → 模型输入
```

---

### 坐标处理

系统内部统一使用：

```text
yzx 坐标系 + mm 单位
```

所有数据自动转换：

```text
xyz → yzx
quat → matrix → rot6D
```

---

### 标准化

使用：

```text
SeqStandardizer
```

作用：

* x → 标准化（输入）
* y → 标准化（标签）


---

## 🚀 训练方法

---

## 1️⃣ Transformer Baseline

运行：

```bash
python -m rigid_flow.train_transformer_with_data_py --config configs/rigid_config.yaml
```

特点：

* 纯序列建模
* 无残差
* 作为 baseline

---

## 2️⃣ Mamba（Coarse Only）

运行：

```bash
python -m rigid_flow.train_mamba_coarse_only --config configs/rigid_config.yaml
```

特点：

* 使用 Mamba 进行时序建模
* 输出直接预测 y9
* 训练稳定、收敛快

损失函数：

```text
L = L1(pos) + L1(rot6D)
```

---

## 3️⃣ Mamba + Flow Matching
运行：

```bash
python -m rigid_flow.train_mamba_with_flow \
    --config configs/rigid_config.yaml \
    --coarse_ckpt runs/mamba_coarse_only/best_model.pt
```

特点：

* 两阶段模型：

  * Stage 1：Mamba 粗预测
  * Stage 2：Flow 学残差
* 精度最高

---

### Flow 模型原理

残差定义：

```text
residual = y_true - y_coarse
```

学习：

```text
v(x,t) ≈ residual
```

训练方式：

```text
xt = (1-t)x0 + t * residual
u  = residual - x0
loss = ||v - u||^2
```


---

## 📈 评估指标

训练过程中自动计算：

### 位置误差

```text
MAE (mm)
L2 (mm)
RMSE (mm)
```

### 旋转误差

```text
Geodesic angle (deg)
```


```text
geodesic_angle(R_pred, R_gt)
```


---

## 🧪 推理

运行：

```bash
python -m rigid_flow.infer_mamba_with_flow_csv \
    --config configs/rigid_config.yaml \
    --ckpt runs/mamba_with_flow/best_model.pt \
    --input_csv data/test.csv \
    --output_dir runs/infer
```

功能：

* CSV → 滑窗 → 模型推理
* 输出：

```text
pred_pose.txt
gt_pose.txt
```


---

## 📦 模型输入输出

### 输入

```text
[B, T, 13]
```

### 输出

```text
[B, 9]
```

---

## ⚠️ 注意事项

### 1️⃣ 坐标系

必须统一：

```text
xyz → yzx
```

否则误差会炸

---

### 2️⃣ rot6D

整个 pipeline 都使用：

```text
rotation → rot6D
```

---

### 3️⃣ relative vs world

支持两种监督：

```text
world（推荐）
relative（相对手背）
```

---

### 4️⃣ Flow 训练

必须：

```text
先训练 coarse
再训练 flow
```

---

### 5️⃣ Windows 注意

如果报错：

```text
No module named triton.backends
```

需要关闭：

```bash
set TORCH_DISABLE_DYNAMO=1
```

---

## 📊 TensorBoard

自动记录：

```bash
tensorboard --logdir runs/
```

包含：

* loss 曲线
* MAE / RMSE
* rot error
* scaler参数
* grad norm

---

## 🎯 推荐训练策略

```text
1. train_mamba_coarse_only
2. train_mamba_with_flow
3. infer
```

---

## 📌 总结

该训练框架实现了：

* 多模态数据融合（动捕 + 传感器）
* 时序建模（Mamba）
* 高精度位姿回归（mm级）
* 残差建模（Flow Matching）

👉 当前性能：

```text
位置误差 ≈ 0.9 mm
旋转误差 ≈ 1.2°
```

---
