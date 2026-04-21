# 动作捕捉 + 数据手套同步采集说明（MoCap + Glove README）

---

## 📌 项目说明

本模块用于**同步采集光学动作捕捉数据（MoCap）与数据手套传感器数据**，生成训练所需的数据集。

系统组成：

```text
动捕系统（LuMo） + 数据手套（串口） + Python采集程序
```

最终输出：

```text
sync_data_with_frame.csv
```

---

## ⚙️ 系统架构

```text
LuMo动捕 → ZMQ → Python
手套传感器 → 串口 → Python
                     ↓
              同步写入CSV
```

---

## 📂 关键文件（只需要关注这些）

```text
采样v2.py                ⭐ 主采集程序（唯一需要运行）
LuMoSDKClient.py        动捕通信接口
LusterFrameStruct_pb2.py 数据结构定义
```

👉 其他文件可以忽略

---

## 🚀 使用步骤（最重要）

---

### 1️⃣ 启动动捕系统

确保：

* 动捕软件已经运行
* 正在广播数据
* IP地址正确（通常为本机）

代码中默认：

```python
ip = "127.0.0.1"
```

（一般不用改）

---

### 2️⃣ 连接数据手套（串口）

打开 `采样v2.py`，找到：

```python
ser = serial.Serial(
    port="COM6",
    baudrate=115200,
```

⚠️ 必须修改：

```text
COM6 → 你的实际串口
```

如何查看：

```bash
设备管理器 → 端口（COM & LPT）
```

---

### 3️⃣ 运行采集

```bash
python 采样v2.py
```

运行后会看到：

```text
开始采集，写入: sync_data_with_frame.csv
按 Ctrl+C 停止
```

---

### 4️⃣ 停止采集

```bash
Ctrl + C
```

数据自动保存

---

## 📊 输出数据格式

文件：

```text
sync_data_with_frame.csv
```

每一行：

```text
frame_idx; marker_id; x; y; z; ch1; ch2; ch3; ch4
```

说明：

| 字段        | 含义      |
| --------- | ------- |
| frame_idx | 帧编号     |
| marker_id | 动捕点ID   |
| x,y,z     | 动捕坐标    |
| ch1~ch4   | 手套传感器数据 |

👉 同一帧多个 marker 会重复 frame_idx

---

## 🧠 同步机制（核心原理）

程序内部实现：

```text
线程1：动捕数据（LuMo）
线程2：串口数据（手套）
线程3：同步写入
```

关键逻辑：

* 每一帧动捕数据
* 搭配“最新一帧串口数据”
* 写入同一行

👉 实现代码见：

---

## 🔌 动捕通信说明

动捕数据通过：

```text
ZMQ (tcp://127.0.0.1:6868)
```

实现：

```python
LuMoSDKClient.Connnect(ip)
frame = LuMoSDKClient.ReceiveData(0)
```

👉 数据结构定义见：

---

## ⚠️ 常见问题（很重要）

---

### ❌ 1. 没有数据

检查：

* 动捕是否在广播
* IP 是否正确
* 防火墙是否拦截

---

### ❌ 2. 串口报错

```text
could not open port
```

解决：

* 检查 COM 号
* 串口是否被占用（关掉Arduino IDE）

---

### ❌ 3. 数据全是0

原因：

* 手套未发送数据
* 波特率不对（必须 115200）

---

### ❌ 4. 卡顿 / 掉帧

原因：

* Python IO瓶颈
* 串口读取过慢

解决：

* 不要开太多 print
* 保持采集程序单独运行

---

## 📌 可修改参数

---

### 串口

```python
port="COM6"
baudrate=115200
```

---

### 输出文件名

```python
out_path = "sync_data_with_frame.csv"
```

---

### 动捕IP

```python
ip = "127.0.0.1"
```

---

## 🎯 推荐使用方式

```text
1. 启动动捕
2. 插好手套（确认串口）
3. 运行采样v2.py
4. 做动作采集
5. Ctrl+C停止
6. 得到CSV
```

---

## 📌 总结

该系统实现：

* 动捕 + 手套数据同步采集
* 毫秒级对齐（基于最近帧）
* 自动保存训练数据

👉 是整个模型训练 pipeline 的数据入口

---
