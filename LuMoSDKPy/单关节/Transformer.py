import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
CSV_IN = "clean_frames_with_angle.csv"
FRAME_COL = "frame_idx"

# Marker IDs
A_ID, B_ID = 102782, 102800
C_ID, D_ID = 102797, 102798  # C 不再作为输入，但 D 仍是要预测的点

# 电压列
V_COLS = ["v1", "v2", "v3", "v4"]

# 时序
SEQ_LEN = 64
STRIDE = 1

# 训练
BATCH_SIZE = 256
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 时间块切分，避免相邻泄漏
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# 特征工程：建议开启（平移不变）
# True: 只用 B-A 的相对坐标 + 电压（输入维度更小更稳）
USE_RELATIVE_TO_A = True


# =========================
# Helpers
# =========================
def col_xyz(pid: int):
    return [f"x_p{pid}", f"y_p{pid}", f"z_p{pid}"]

def zscore_fit(x: np.ndarray):
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True) + 1e-8
    return mu, sd

def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd

def split_by_blocks(n: int, val_ratio: float, test_ratio: float, seed: int = 42):
    """
    按时间块随机抽块做val/test，避免泄漏
    """
    rng = np.random.default_rng(seed)
    block = max(300, n // 200)  # 你也可以按100fps改成固定1000更直观
    n_blocks = int(np.ceil(n / block))
    ids = np.arange(n_blocks)
    rng.shuffle(ids)

    n_test = max(1, int(round(n_blocks * test_ratio)))
    n_val  = max(1, int(round(n_blocks * val_ratio)))

    test_blocks = set(ids[:n_test].tolist())
    val_blocks  = set(ids[n_test:n_test + n_val].tolist())

    blk = (np.arange(n) // block)
    is_test = np.array([b in test_blocks for b in blk])
    is_val  = np.array([b in val_blocks for b in blk])
    is_train = ~(is_test | is_val)
    return is_train, is_val, is_test

class SeqDataset(Dataset):
    """
    用过去 L 帧预测当前帧
    X: [N, Din], Y: [N, Dout]
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, seq_len: int, stride: int):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.L = int(seq_len)
        self.stride = int(stride)
        n = len(X)
        self.idxs = np.arange(self.L - 1, n, self.stride)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.X[t - self.L + 1: t + 1]  # [L, Din]
        y = self.Y[t]                      # [Dout]
        return torch.from_numpy(x), torch.from_numpy(y)

# =========================
# Model
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class TransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 256, dropout: float = 0.1, out_dim: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.enc(h)
        return self.head(h[:, -1, :])  # 用最后一帧表征预测当前帧输出

# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_IN).sort_values(FRAME_COL).reset_index(drop=True)

    # 只要求 A、B、D + 电压
    need_cols = [FRAME_COL] + V_COLS + col_xyz(A_ID) + col_xyz(B_ID) + col_xyz(D_ID)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV 缺少列: {missing}")

    # 取数据
    A = df[col_xyz(A_ID)].to_numpy(float)
    B = df[col_xyz(B_ID)].to_numpy(float)
    D = df[col_xyz(D_ID)].to_numpy(float)
    V = df[V_COLS].to_numpy(float)

    # ======== 只用 A、B + V 作为输入，预测 D ========
    if USE_RELATIVE_TO_A:
        # 输入：B相对A + 电压（A自己不喂给网络）
        B_rel = B - A
        X = np.hstack([B_rel, V])     # 3 + 4 = 7
        Y = D - A                     # 预测 D 相对 A（3）
        target_is_relative = True
    else:
        # 输入：A、B 绝对坐标 + 电压
        X = np.hstack([A, B, V])      # 3 + 3 + 4 = 10
        Y = D                         # 预测 D 绝对坐标
        target_is_relative = False

    # NaN过滤
    m = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    df = df[m].copy().reset_index(drop=True)
    X = X[m]
    Y = Y[m]

    n = len(df)
    print(f"[Info] N={n}, X_dim={X.shape[1]}, target_relative={target_is_relative}")

    # split
    is_train, is_val, is_test = split_by_blocks(n, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
    X_tr, Y_tr = X[is_train], Y[is_train]
    X_va, Y_va = X[is_val], Y[is_val]
    X_te, Y_te = X[is_test], Y[is_test]

    # 标准化（只用train统计量）
    mu_x, sd_x = zscore_fit(X_tr)
    mu_y, sd_y = zscore_fit(Y_tr)

    X_tr = zscore_apply(X_tr, mu_x, sd_x)
    X_va = zscore_apply(X_va, mu_x, sd_x)
    X_te = zscore_apply(X_te, mu_x, sd_x)

    Y_tr_n = zscore_apply(Y_tr, mu_y, sd_y)
    Y_va_n = zscore_apply(Y_va, mu_y, sd_y)
    Y_te_n = zscore_apply(Y_te, mu_y, sd_y)

    # dataset
    ds_tr = SeqDataset(X_tr, Y_tr_n, SEQ_LEN, STRIDE)
    ds_va = SeqDataset(X_va, Y_va_n, SEQ_LEN, STRIDE)
    ds_te = SeqDataset(X_te, Y_te_n, SEQ_LEN, STRIDE)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = TransformerRegressor(in_dim=X.shape[1], d_model=128, nhead=8, num_layers=4, dim_ff=256, dropout=0.1, out_dim=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_main = nn.SmoothL1Loss(beta=1.0)

    def eval_mae(dl):
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(DEVICE)
                pred = model(x).cpu().numpy()
                ys.append(y.numpy())
                yh.append(pred)
        ys = np.concatenate(ys, axis=0)
        yh = np.concatenate(yh, axis=0)

        ys_real = ys * sd_y + mu_y
        yh_real = yh * sd_y + mu_y

        mae_xyz = np.mean(np.abs(ys_real - yh_real), axis=0)
        mae_euc = np.mean(np.linalg.norm(ys_real - yh_real, axis=1))
        return mae_xyz, float(mae_euc)

    best_val = 1e18
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        cnt = 0

        for x, y in dl_tr:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_main(pred, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * x.size(0)
            cnt += x.size(0)

        tr_loss = total / max(cnt, 1)

        va_mae_xyz, va_mae_euc = eval_mae(dl_va)
        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_MAE_xyz={va_mae_xyz} | val_MAE_euc={va_mae_euc:.4f}")

        if va_mae_euc < best_val:
            best_val = va_mae_euc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # test
    model.load_state_dict(best_state)
    te_mae_xyz, te_mae_euc = eval_mae(dl_te)
    print("\n==== Test ====")
    print(f"MAE_xyz = {te_mae_xyz}")
    print(f"MAE_euc = {te_mae_euc:.4f} (same unit as mocap coordinates)")

    # 可视化一段预测（看时序）
    model.eval()
    with torch.no_grad():
        x0, y0 = next(iter(dl_te))
        pred0 = model(x0.to(DEVICE)).cpu().numpy()
        y0 = y0.numpy()

    y0_real = y0 * sd_y + mu_y
    p0_real = pred0 * sd_y + mu_y

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(y0_real[:1000, 0], label="GT Dx (relative)" if target_is_relative else "GT Dx")
    plt.plot(p0_real[:1000, 0], label="Pred Dx")
    plt.title("Example segment (Dx)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
