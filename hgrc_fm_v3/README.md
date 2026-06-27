# Model Training / 模型训练

## 中文说明

本目录包含中指末端位姿重建模型的训练、评估和推理代码。模型输入为时间窗口内的手背参考位姿和三路光纤传感信号，输出为指尖刚体中心的三维位置与 6D 旋转表示。

### 数据格式

```text
input  [B, T, 12] = hand-back pose 9D + fiber channels 3D
target [B, 9]     = fingertip position 3D + fingertip rotation 6D
```

早期采集代码中可能保留四通道采集卡的兼容逻辑，但训练管线使用 MCP、PIP、DIP 对应的三路有效光纤通道。

### 主要文件

```text
rigid_flow/data.py                         Dataset and normalization
rigid_flow/geometry.py                     Rotation, quaternion, rot6D utilities
rigid_flow/models.py                       Transformer, Mamba, Flow models
rigid_flow/train_transformer_with_data_py.py
rigid_flow/train_mamba_coarse_only.py
rigid_flow/train_mamba_with_flow.py
rigid_flow/train_recent_baselines.py       PatchTST, ModernTCN, large BiLSTM
rigid_flow/infer_mamba_with_flow_csv.py    CSV inference
rigid_flow/evaluate_recalibration.py       Recalibration/generalization tests
```

### 训练示例

```bash
pip install -r requirements.txt
python -m rigid_flow.train_mamba_coarse_only --config configs/rigid_config.yaml
python -m rigid_flow.train_mamba_with_flow --config configs/rigid_config.yaml --coarse_ckpt runs/mamba_coarse_only/best_model.pt
```

Baseline:

```bash
python -m rigid_flow.train_transformer_with_data_py --config configs/rigid_config.yaml
python -m rigid_flow.train_recent_baselines --config configs/rigid_config.yaml --model patchtst --epochs 20
python -m rigid_flow.train_recent_baselines --config configs/rigid_config.yaml --model moderntcn --epochs 20
python -m rigid_flow.train_recent_baselines --config configs/rigid_config.yaml --model bilstm_large --epochs 20
```

### 推理

```bash
python -m rigid_flow.infer_mamba_with_flow_csv \
  --config configs/rigid_config.yaml \
  --ckpt runs/mamba_with_flow/best_model.pt \
  --input_csv data/test.csv \
  --output_dir runs/infer_mamba_with_flow_csv
```

`data/` 与 `runs/` 不随仓库提交。请根据预处理流程生成 CSV，或自行下载数据后放入对应路径。

## English

This module contains training, evaluation, and inference code for middle-fingertip pose reconstruction. The model takes a temporal window of hand-back reference pose and three fiber-sensor channels, and predicts the fingertip rigid-body center position and 6D rotation.

Some legacy acquisition scripts contain four-channel compatibility code because the acquisition board had four analog channels. The cleaned training pipeline uses the three active channels for MCP, PIP, and DIP joints.

Recommended workflow:

1. Prepare CSV files under `data/`.
2. Train the Mamba coarse predictor.
3. Train the Flow Matching residual refiner from the coarse checkpoint.
4. Run CSV inference and visualize the exported pose files.

Checkpoints and generated outputs are saved under `runs/`, which is ignored by Git.
