# 数据采集与预处理流程说明

## 📌 功能说明

本模块用于完成数据手套实验中的**多模态数据采集与预处理**，包括：

* 光学动捕 marker 数据采集
* 光纤传感器 / 商用手套数据同步
* marker 跟踪与清洗
* 刚体位姿计算
* 数据格式转换与切分

---

## 📂 数据采集流程

### 1️⃣ 同步采集动捕 + 传感器数据

运行：

```bash
python 采样v2.py
```

👉 功能：

* 从动捕系统读取 marker 数据
* 从串口读取光纤传感器数据
* 按帧同步写入 CSV

👉 输出：

```
sync_data_with_frame.csv
```

👉 数据格式：

```
frame_idx; marker_id; x; y; z; ch1; ch2; ch3; ch4
```

---


## 📊 数据预处理流程

### 2️⃣ marker 跟踪与清洗

运行：

```bash
python 标记点跟踪.py
```

或（带关节版本）：

```bash
python 标记点跟踪 关节.py
```

👉 功能：

* 跟踪：

  * 手背刚体 4 点
  * 指尖刚体 4 点
  * （可选）PIP / DIP 关节
* 自动处理：

  * marker ID 变化
  * 丢点 / 跳变
  * 异常帧检测

👉 输出：

* `clean_glove_one_row_per_frame.csv`
* `abnormal_frames_log.csv`
* `id_change_log.csv`

---

### 3️⃣ 帧裁剪（去掉无效段）

运行：

```bash
python 帧裁剪.py
```

👉 功能：

* 截取有效数据区间
* 重新编号 frame_idx

---

### 4️⃣ 刚体位姿计算

运行：

```bash
python 计算刚体中心位置与四元数姿态.py
```

👉 功能：

* 根据 4 个 marker 计算：

  * 刚体中心位置
  * 四元数姿态

👉 输出：

* 手背刚体 pose
* 指尖刚体 pose

（核心逻辑：四点构建坐标系 + 正交化）

---

### 5️⃣ 数据格式转换（用于模型训练）

运行：

```bash
python 数据转换脚本.py
```

👉 功能：

* 将 CSV 转为模型可用的 TXT 格式
* 自动处理：

  * yzx ↔ xyz 坐标转换
  * 四元数计算

👉 输出格式：

```
back_pose + tip_pose + sensor
```

---

### 6️⃣ 数据集划分

运行：

```bash
python split_dataset.py
```

👉 功能：

* 划分 train / val / test 数据集

👉 输出：

```
data/train.csv
data/val.csv
data/test.csv
```

---

## 📈 可视化工具（可选）

### marker 可视化

```bash
python 数据集可视化.py
```

或带关节：

```bash
python 数据集可视化 带关节.py
```

👉 功能：

* 查看 marker / 刚体结构
* 播放数据
* 检查 tracking 是否正确

---

### 双刚体位姿可视化

```bash
python 可视化刚体中心点.py
```

👉 功能：

* 显示手背 + 指尖刚体
* 显示姿态（四元数坐标轴）

---

## 📌 数据格式说明

### 原始采集数据

```
frame_idx; marker_id; x; y; z; ch1; ch2; ch3; ch4
```

---

### 清洗后数据（一帧一行）

包含：

* 手背 4 点
* 指尖 4 点
* （可选）关节点
* 传感器数据

---

### 模型输入数据

```
back_pose + tip_pose + sensor
```

---

## ⚠️ 注意事项

* 动捕坐标为 **yzx 语义**，需统一转换
* marker ID 可能变化，必须做跟踪
* 建议先检查 `abnormal_frames_log.csv`
* 数据采集时尽量保证：

  * marker 不遮挡
  * 手套稳定佩戴

---

## 📦 依赖

```bash
pip install numpy pandas matplotlib pyserial
```

---

## 📌 总流程

```
采集 → 清洗 → 裁剪 → 位姿计算 → 格式转换 → 数据集划分
```

---

## 🎯 用途

* 构建训练数据集
* 多模态标定
* 手指位姿重建模型训练
