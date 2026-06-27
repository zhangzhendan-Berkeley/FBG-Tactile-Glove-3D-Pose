import sys
from collections import deque

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import serial
from serial.tools import list_ports
import threading
import time


def parse_frame(line: bytes):
    """
    兼容你 C++ 的解析逻辑（非常像你那套 string2float）：
    - 一帧（不含 \r\n）里需要包含: z1 a ... b ... c ... d ... e ... f ... g ... h ... i
    - a~i 是分隔符，a->b, b->c ... h->i 之间分别是 8 个 float
    - 只有以 'z1' 开头才认为有效
    """
    s = line.decode("ascii", errors="ignore").strip()
    if len(s) < 2 or not (s[0] == "z" and s[1] == "1"):
        return None

    # 找分隔符位置
    idx = {}
    for ch in "abcdefghi":
        p = s.find(ch)
        if p < 0:
            return None
        idx[ch] = p

    # 按顺序
    pos = [idx[ch] for ch in "abcdefghi"]
    if pos != sorted(pos):
        return None

    vals = []
    for a, b in zip("abcdefgh", "bcdefghi"):
        seg = s[idx[a] + 1: idx[b]].strip()
        if not seg:
            return None
        try:
            vals.append(float(seg))
        except ValueError:
            return None

    return vals if len(vals) == 8 else None


class SerialWorker(QtCore.QObject):
    new_values = QtCore.Signal(list)  # 8 floats
    status = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._ser = None
        self._running = False
        self._buf = bytearray()

    @QtCore.Slot(str)
    def open(self, port_name: str):
        self.close()
        try:
            self._ser = serial.Serial(
                port=port_name,
                baudrate=115200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.00,
            )
            self._running = True
            self.status.emit(f"已打开串口: {port_name}")
        except Exception as e:
            self._ser = None
            self._running = False
            self.status.emit(f"打开失败: {e}")

    @QtCore.Slot()
    def close(self):
        self._running = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._buf.clear()
        self.status.emit("串口已关闭")

    @QtCore.Slot()
    def loop(self):
        while True:
            if not self._running or self._ser is None:
                QtCore.QThread.msleep(50)
                continue

            try:
                n = self._ser.in_waiting
                if n <= 0:
                    QtCore.QThread.msleep(1)
                    continue
                data = self._ser.read(n)
            except Exception as e:
                self.status.emit(f"读串口异常: {e}")
                self.close()
                continue

            self._buf.extend(data)

            # 按 \r\n 分帧
            while True:
                p = self._buf.find(b"\r\n")
                if p < 0:
                    break
                frame = bytes(self._buf[:p])
                del self._buf[:p + 2]

                # DEBUG: 打印原始帧（前几条）
                # print("RAW:", frame[:200])  # 只打印前200字节，防止刷屏

                vals = parse_frame(frame)
                if vals is None:
                    # print("PARSE FAIL")  # 解析失败会一直出现
                    continue
                else:
                    print("OK:", vals)  # 解析成功会打印8个数
                    self.new_values.emit(vals)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("8路电压实时曲线（Python）")
        self.resize(1100, 750)

        # 顶部控制区
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top)
        self.port_combo = QtWidgets.QComboBox()
        self.btn_scan = QtWidgets.QPushButton("扫描串口")
        self.btn_open = QtWidgets.QPushButton("打开串口")
        self.btn_pause = QtWidgets.QPushButton("暂停")
        self.lbl_status = QtWidgets.QLabel("未连接")
        top_layout.addWidget(QtWidgets.QLabel("串口:"))
        top_layout.addWidget(self.port_combo, 1)
        top_layout.addWidget(self.btn_scan)
        top_layout.addWidget(self.btn_open)
        top_layout.addWidget(self.btn_pause)
        top_layout.addWidget(self.lbl_status, 2)

        # 曲线区
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Voltage (V)")
        self.plot.setLabel("bottom", "Samples")
        self.plot.setYRange(0, 1)
        self.plot.setXRange(0, 1000)

        # 右侧 checkbox
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.checks = []
        for i in range(8):
            cb = QtWidgets.QCheckBox(f"CH{i+1}")
            cb.setChecked(i == 0)
            cb.stateChanged.connect(self.refresh_visibility)
            self.checks.append(cb)
            right_layout.addWidget(cb)
        right_layout.addStretch(1)

        # 主布局
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QHBoxLayout(center)
        center_layout.addWidget(self.plot, 5)
        center_layout.addWidget(right, 1)

        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.addWidget(top)
        root_layout.addWidget(center, 1)
        self.setCentralWidget(root)

        # 数据缓冲：每路 100 点
        self.max_points = 1000
        self.y = [deque([0.0] * self.max_points, maxlen=self.max_points) for _ in range(8)]
        self.latest = [0.0] * 8

        # 8 条曲线
        pens = [
            pg.mkPen("orange", width=2),
            pg.mkPen("r", width=2),
            pg.mkPen("g", width=2),
            pg.mkPen("b", width=2),
            pg.mkPen("c", width=2),
            pg.mkPen("m", width=2),
            pg.mkPen("y", width=2),
            pg.mkPen("w", width=2),
        ]
        self.curves = [self.plot.plot(list(range(self.max_points)), list(self.y[i]), pen=pens[i]) for i in range(8)]

        # 定时刷新（10ms）
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.redraw)
        self.timer.start()
        self.paused = False

        # 串口线程
        self.worker = SerialWorker()
        self.thread = QtCore.QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.loop)
        self.worker.new_values.connect(self.on_new_values)
        self.worker.status.connect(self.on_status)
        self.thread.start()

        # 按钮
        self.btn_scan.clicked.connect(self.scan_ports)
        self.btn_open.clicked.connect(self.toggle_open)
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.scan_ports()
        self.xs = list(range(self.max_points))
        # ===== 性能统计 =====
        self._t_stat0 = time.perf_counter()
        self._redraw_cnt = 0
        self._onvals_cnt = 0

        self._dt_redraw_sum = 0.0
        self._dt_redraw_max = 0.0

        self._dt_list_sum = 0.0
        self._dt_list_max = 0.0

        self._dt_setdata_sum = 0.0
        self._dt_setdata_max = 0.0

        self._dt_onvals_sum = 0.0
        self._dt_onvals_max = 0.0


    def closeEvent(self, event):
        try:
            self.worker.close()
        except Exception:
            pass
        event.accept()

    @QtCore.Slot()
    def scan_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in list_ports.comports()]
        self.port_combo.addItems(ports)
        if not ports:
            self.on_status("未发现串口")

    @QtCore.Slot()
    def toggle_open(self):
        if self.btn_open.text() == "打开串口":
            port = self.port_combo.currentText().strip()
            if not port:
                self.on_status("请选择串口")
                return
            self.worker.open(port)
            self.btn_open.setText("关闭串口")
        else:
            self.worker.close()
            self.btn_open.setText("打开串口")

    @QtCore.Slot()
    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.setText("继续" if self.paused else "暂停")

    @QtCore.Slot(list)
    def on_new_values(self, vals):
        t0 = time.perf_counter()

        self.latest = vals
        for i in range(8):
            self.y[i].append(vals[i])

        dt = time.perf_counter() - t0
        self._onvals_cnt += 1
        self._dt_onvals_sum += dt
        if dt > self._dt_onvals_max:
            self._dt_onvals_max = dt

    @QtCore.Slot(str)
    def on_status(self, msg: str):
        self.lbl_status.setText(msg)

    @QtCore.Slot()
    def refresh_visibility(self):
        for i, cb in enumerate(self.checks):
            self.curves[i].setVisible(cb.isChecked())

    @QtCore.Slot()
    def redraw(self):
        if self.paused:
            return

        t0 = time.perf_counter()

        xs = self.xs

        # 统计本次 redraw 中 list() 和 setData() 的耗时
        t_list_sum = 0.0
        t_set_sum = 0.0

        for i in range(8):
            if self.checks[i].isChecked():
                t1 = time.perf_counter()
                yi = list(self.y[i])  # <-- 怀疑点1：Python拷贝
                t2 = time.perf_counter()
                self.curves[i].setData(xs, yi)  # <-- 怀疑点2：绘制/转换
                t3 = time.perf_counter()

                t_list_sum += (t2 - t1)
                t_set_sum += (t3 - t2)

        dt = time.perf_counter() - t0

        self._redraw_cnt += 1
        self._dt_redraw_sum += dt
        self._dt_list_sum += t_list_sum
        self._dt_setdata_sum += t_set_sum

        if dt > self._dt_redraw_max:
            self._dt_redraw_max = dt
        if t_list_sum > self._dt_list_max:
            self._dt_list_max = t_list_sum
        if t_set_sum > self._dt_setdata_max:
            self._dt_setdata_max = t_set_sum

        # 每 1 秒打印一次统计
        now = time.perf_counter()
        if now - self._t_stat0 >= 1.0:
            sec = now - self._t_stat0

            redraw_fps = self._redraw_cnt / sec
            onvals_fps = self._onvals_cnt / sec

            avg_redraw_ms = (self._dt_redraw_sum / max(1, self._redraw_cnt)) * 1000
            avg_list_ms = (self._dt_list_sum / max(1, self._redraw_cnt)) * 1000
            avg_set_ms = (self._dt_setdata_sum / max(1, self._redraw_cnt)) * 1000

            # print(
            #     f"[1s] redraw_fps={redraw_fps:6.1f}  onvals_fps={onvals_fps:6.1f} | "
            #     f"redraw avg={avg_redraw_ms:6.2f}ms max={self._dt_redraw_max * 1000:6.2f}ms | "
            #     f"list avg={avg_list_ms:6.2f}ms max={self._dt_list_max * 1000:6.2f}ms | "
            #     f"setData avg={avg_set_ms:6.2f}ms max={self._dt_setdata_max * 1000:6.2f}ms | "
            #     f"on_new avg={(self._dt_onvals_sum / max(1, self._onvals_cnt)) * 1000:6.3f}ms max={self._dt_onvals_max * 1000:6.3f}ms"
            # )

            # reset window
            self._t_stat0 = now
            self._redraw_cnt = 0
            self._onvals_cnt = 0
            self._dt_redraw_sum = 0.0
            self._dt_redraw_max = 0.0
            self._dt_list_sum = 0.0
            self._dt_list_max = 0.0
            self._dt_setdata_sum = 0.0
            self._dt_setdata_max = 0.0
            self._dt_onvals_sum = 0.0
            self._dt_onvals_max = 0.0


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
