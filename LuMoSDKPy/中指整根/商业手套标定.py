# -*- coding: utf-8 -*-
import time
import csv
import subprocess
import threading
import queue
import LuMoSDKClient as LuMoSDKClient


# =========================================================
# 从 C++ bridge 程序 stdout 中读取最新一帧 sdata
# =========================================================
class WiseGloveBridgeReader:
    def __init__(self, exe_path):
        self.exe_path = exe_path
        self.proc = None
        self.q = queue.Queue()
        self.latest_sdata = None
        self.num_sensor = None
        self._stop = False
        self._thread = None

    def start(self):
        self.proc = subprocess.Popen(
            [self.exe_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

        self._err_thread = threading.Thread(target=self._stderr_thread, daemon=True)
        self._err_thread.start()

    def _reader_thread(self):
        while not self._stop:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.001)
                continue

            line = line.strip()
            if not line.startswith("DATA"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                vals = [int(x) for x in parts[1:]]
            except ValueError:
                continue

            self.latest_sdata = vals
            self.num_sensor = len(vals)

    def _stderr_thread(self):
        while not self._stop:
            line = self.proc.stderr.readline()
            if not line:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.001)
                continue
            print("[WiseGloveBridge]", line.strip())

    def read_latest(self):
        return self.latest_sdata

    def close(self):
        self._stop = True
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass


# =========================================================
# 主程序：LuMo 拉帧 + 商用手套原始值 + 写 CSV
# =========================================================
def main():
    # 1) 启动商用手套 bridge
    glove = WiseGloveBridgeReader(exe_path="wiseglove_stream.exe")
    glove.start()

    print("等待商用手套数据...")
    t0 = time.time()
    while glove.read_latest() is None:
        if time.time() - t0 > 10:
            raise RuntimeError("10 秒内没有收到商用手套数据，请检查 exe / dll / 手套连接")
        time.sleep(0.01)

    print(f"已收到商用手套数据，通道数 = {glove.num_sensor}")

    # 2) 连接 LuMo
    ip = "127.0.0.1"
    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(ip)

    # 3) 输出文件
    out_path = "sync_lumo_wiseglove_raw.csv"
    frame_idx = 0

    print(f"开始采集，写入: {out_path}")
    print("按 Ctrl+C 停止。")

    try:
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            header = ["frame_idx", "marker_id", "x", "y", "z"]
            header += [f"sdata_{i+1}" for i in range(glove.num_sensor)]
            writer.writerow(header)

            while True:
                # 当前最新商用手套原始值
                latest_sdata = glove.read_latest()
                if latest_sdata is None:
                    time.sleep(0.001)
                    continue

                # 拉一帧 LuMo
                frame = LuMoSDKClient.ReceiveData(0)
                if frame is None:
                    time.sleep(0.001)
                    continue

                if not hasattr(frame, "markers") or frame.markers is None:
                    frame_idx += 1
                    continue

                # 同一帧内所有 marker 共用同一组 latest_sdata
                for marker in frame.markers:
                    marker_id = marker.Id
                    x, y, z = marker.X, marker.Y, marker.Z

                    row = [
                        frame_idx,
                        marker_id,
                        f"{x:.5f}",
                        f"{y:.5f}",
                        f"{z:.5f}",
                    ] + latest_sdata

                    writer.writerow(row)

                frame_idx += 1

                if frame_idx % 200 == 0:
                    print(f"已采集 {frame_idx} 帧")

    finally:
        glove.close()
        print("采集结束。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("停止采集中。")