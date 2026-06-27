import time
import serial
import LuMoSDKClient as LuMoSDKClient


# =========================
# 采集板串口解析：z1 a..i -> 8 floats (按 \r\n 分帧)
# =========================
def parse_frame(line: bytes):
    """
    一帧（不含 \r\n）格式：
    z1 a<val0> b<val1> c<val2> d<val3> e<val4> f<val5> g<val6> h<val7> i
    只要以 'z1' 开头并且 a..i 分隔符齐全、顺序正确，就解析出 8 个 float
    """
    s = line.decode("ascii", errors="ignore").strip()
    if len(s) < 2 or not (s[0] == "z" and s[1] == "1"):
        return None

    idx = {}
    for ch in "abcdefghi":
        p = s.find(ch)
        if p < 0:
            return None
        idx[ch] = p

    pos = [idx[ch] for ch in "abcdefghi"]
    if pos != sorted(pos):
        return None

    vals = []
    for a, b in zip("abcdefgh", "bcdefghi"):
        seg = s[idx[a] + 1 : idx[b]].strip()
        if not seg:
            return None
        try:
            vals.append(float(seg))
        except ValueError:
            return None

    return vals if len(vals) == 8 else None


def try_parse_latest_8ch(rx_buf: bytearray):
    """
    从累积 buffer 中按 \r\n 分帧，解析出“最新一帧”的 8 路 float。
    返回 (latest_vals_or_None, new_buf)
    """
    latest = None

    while True:
        p = rx_buf.find(b"\r\n")
        if p < 0:
            if len(rx_buf) > 8192:
                rx_buf = rx_buf[-1024:]
            break

        frame = bytes(rx_buf[:p])
        del rx_buf[:p + 2]

        vals = parse_frame(frame)
        if vals is not None:
            latest = vals

    return latest, rx_buf


# =========================
# 主程序：LuMo 拉帧 + 串口非阻塞读 + 写文件
# =========================
def main():
    # 1) 连接 LuMo
    ip = "127.0.0.1"
    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(ip)

    # 2) 打开串口（非阻塞）
    ser = serial.Serial(
        port="COM6",
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.0,
    )
    rx_buf = bytearray()

    # 3) 输出文件
    out_path = "sync_data_with_frame.csv"
    latest_8ch = [0.0] * 8
    frame_idx = 0

    print(f"开始采集，写入: {out_path}")
    print("按 Ctrl+C 停止。")

    try:
        with open(out_path, "w", buffering=1, encoding="utf-8") as f:
            # 写表头
            f.write("frame_idx; marker_id; x; y; z; ch1; ch2; ch3; ch4\n")

            while True:
                # ---- 先读串口（更新最新 8ch）----
                try:
                    n = ser.in_waiting
                    if n:
                        rx_buf += ser.read(n)
                        parsed, rx_buf = try_parse_latest_8ch(rx_buf)
                        if parsed is not None:
                            latest_8ch = parsed
                except Exception as e:
                    print(f"[Serial] 读串口异常: {e}")
                    time.sleep(0.01)

                # ---- 拉一帧动捕 ----
                frame = LuMoSDKClient.ReceiveData(0)
                if frame is None:
                    time.sleep(0.001)
                    continue

                # 只写前 4 路电压
                chs = latest_8ch[0:4]

                # 如果这一帧没有 marker，也可以选择跳过
                if not hasattr(frame, "markers") or frame.markers is None:
                    frame_idx += 1
                    continue

                # 同一帧内所有 marker 共用同一个 frame_idx 和同一时刻 latest_8ch
                for marker in frame.markers:
                    marker_id = marker.Id
                    x, y, z = marker.X, marker.Y, marker.Z

                    line = (
                        f"{frame_idx}; "
                        f"{marker_id}; "
                        f"{x:.5f}; {y:.5f}; {z:.5f}; "
                        f"{chs[0]:.6f}; {chs[1]:.6f}; {chs[2]:.6f}; {chs[3]:.6f}\n"
                    )
                    f.write(line)

                # 一次 ReceiveData 成功处理完，就认为完成 1 帧
                frame_idx += 1

                if frame_idx % 200 == 0:
                    print(f"已采集 {frame_idx} 帧")

    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("串口已关闭。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("停止采集中。")