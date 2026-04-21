import tkinter as tk
import queue
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.collections import LineCollection

# 假设你已经有了 NatNetClient 相关的代码文件，这里只做接口调用示范
# from NatNetClient_mmy import NatNetClient
import LuMoSDKClient as LuMoSDKClient
import pickle


# ---------------------------- 工具函数 ----------------------------
test_number = 3
name = "mmy_biaoding_8_29"
save_name = name + "_test_number_" + str(test_number) + "_truth_and_force_2D_test.pkl"

def angle_between_vectors(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def limit_rotation(prev_normal, curr_normal, max_angle_deg=5):
    angle = angle_between_vectors(prev_normal, curr_normal)
    if angle <= max_angle_deg:
        return curr_normal / np.linalg.norm(curr_normal)

    axis = np.cross(prev_normal, curr_normal)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return prev_normal / np.linalg.norm(prev_normal)
    axis /= axis_norm

    max_angle_rad = np.radians(max_angle_deg)

    def rodrigues_rotate(v, k, theta):
        v = v / np.linalg.norm(v)
        return (v * np.cos(theta) +
                np.cross(k, v) * np.sin(theta) +
                k * np.dot(k, v) * (1 - np.cos(theta)))

    new_normal = rodrigues_rotate(prev_normal, axis, max_angle_rad)
    return new_normal / np.linalg.norm(new_normal)


class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.state = None

    def filter(self, new_data):
        new_data = np.array(new_data)
        if self.state is None:
            self.state = new_data
        else:
            self.state = self.alpha * new_data + (1 - self.alpha) * self.state
        return self.state


# ---------------------------- 画笔几何参数 ----------------------------

BRUSH_LEN = 0.21
TIP_LEN = 0.04
SHAFT_RADIUS = 0.002
RESOLUTION = 32

SHAFT_COLOR = "#8B5A2B"
TIP_COLOR = "#604333"
PATH_COLOR = "#20bf6b"

data_queue = queue.Queue()
serial_raw_queue = queue.Queue()

# ---------------------------- 几何计算 ----------------------------

def compute_brush_geometry(points):
    """
    输入两个点（根部和笔尖），判断哪个是笔尖( z小的)，并计算笔杆和笔尖位置
    """
    pts = np.array(points)
    if pts.shape[0] != 2:
        raise ValueError("需要且仅需要两个点")

    # 根据z判断笔尖(小z为笔尖)
    if pts[0, 2] < pts[1, 2]:
        tip = pts[0]
        root = pts[1]
    else:
        tip = pts[1]
        root = pts[0]

    axis = tip - root
    length = np.linalg.norm(axis)
    if length < 1e-6:
        normal = np.array([0, 0, 1])
    else:
        normal = axis / length

    # shaft_bottom在root和tip之间，稍微向tip延伸TIP_LEN长度
    shaft_bottom = root + normal * (length - TIP_LEN)
    tip_point = tip

    return root, normal, shaft_bottom, tip_point


def make_orthonormal_basis(n):
    n = n / np.linalg.norm(n)
    tmp = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = tmp - np.dot(tmp, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def _surface(start, n, r_mat, z_mat, theta_mat):
    e1, e2 = make_orthonormal_basis(n)
    cosT, sinT = np.cos(theta_mat), np.sin(theta_mat)
    X = start[0] + n[0] * z_mat + r_mat * (e1[0] * cosT + e2[0] * sinT)
    Y = start[1] + n[1] * z_mat + r_mat * (e1[1] * cosT + e2[1] * sinT)
    Z = start[2] + n[2] * z_mat + r_mat * (e1[2] * cosT + e2[2] * sinT)
    return X, Y, Z



# ---------------------------- GUI类 ----------------------------

class BrushGUI:
    def __init__(self, master: tk.Tk):
        master.title("Interactive 3D Brush – Minimal")
        self.master = master
        self.path_data_name = "long_path_data.pkl"
        self.start_time = time.time()
        self.last_drawing = False  # 是否上一次是“正在写”
        self.last_mocap_print_time = 0
        self.mocap_print_interval = 0.1  # 秒
        self.last_serial_print_time = 0
        self.serial_print_interval = 0.1  # 秒
        self.last_serial_truth_force_print_time = 0
        self.serial_truth_force_print_interval = 0.1  # 秒
        self.historical_force_arrows = []  # 存储 (x0, y0, dx, dy)
        self.arrow_counter = 0
        self.path_points = []
        self._surfaces = []
        self.filters = [EMAFilter(alpha=0.3) for _ in range(2)]
        self.current_points = None

        
    def _draw_scene(self, points):
        if points is None or len(points) != 6:
            return

        sorted_points = sorted(points, key=lambda p: p[2])
        paper = np.array(sorted_points[0])
        tip = np.array(sorted_points[1])
        plane_points = [np.array(p) for p in sorted_points[2:6]]

        distances = [np.linalg.norm(tip - p) for p in plane_points]
        min_index = distances.index(min(distances))
        closest_point = plane_points[min_index]
        end = closest_point

        plane_origin = np.mean(plane_points, axis=0)
        x_dir = closest_point - plane_origin
        x_len = np.linalg.norm(x_dir)
        x_dir = x_dir / x_len if x_len > 1e-6 else np.array([1, 0, 0])
        

        filtered_tip, filtered_end = [f.filter(p) for f, p in zip(self.filters, [tip, end])]
        axis = filtered_end - filtered_tip
        length = np.linalg.norm(axis)
        normal = axis / length if length > 1e-6 else np.array([0, 0, -1])
        if np.dot(normal, [0, 0, -1]) < 0:
            normal = -normal
        z_dir = -normal

        y_dir = np.cross(z_dir, x_dir)
        y_len = np.linalg.norm(y_dir)
        y_dir = y_dir / y_len if y_len > 1e-6 else np.array([0, 1, 0])

        shaft_bottom = filtered_tip + normal * TIP_LEN

        tip_to_paper_distance = abs(shaft_bottom[2] - paper[2])
        is_drawing = tip_to_paper_distance < 0.016

        # ----------- 画力的 XY 分量箭头 -----------

        # if hasattr(self, "force_arrow") and self.force_arrow is not None:
        #     self.force_arrow.remove()
        #     self.force_arrow = None

        # force_vec = np.array([0, 0, 0])
        if not hasattr(self, 'path_forces'):
            self.path_forces = []
        if not hasattr(self, 'path_rot'):
            self.path_rot = []
        if not hasattr(self, 'truth_forces'):
            self.truth_forces = []
        if not hasattr(self, 'arrow_counter'):
            self.arrow_counter = 0
        if not hasattr(self, 'historical_force_arrows'):
            self.historical_force_arrows = []

        local_to_global_mat = np.column_stack((x_dir, y_dir, z_dir))
        if hasattr(self, 'latest_force'):
            f1, f2, f3 = self.latest_force
            force_vec = f1 * x_dir + f2 * y_dir + f3 * z_dir
            
          
        if hasattr(self, 'latest_truth_force'):
            truth_record = self.latest_truth_force

        if hasattr(self, 'latest_raw_data'):
            raw_data = self.latest_raw_data
        # ----------- 记录轨迹（XY） -----------

        if hasattr(self, 'latest_truth_force') and hasattr(self, 'latest_raw_data') :
            self.path_forces.append(raw_data[:].copy())     # 仅记录 XY 力
            self.truth_forces.append(truth_record[:].copy())
            self.path_rot.append(local_to_global_mat.copy())
            print("sensor_force:", raw_data)
            print("truth_force:", truth_record)
            print("rot:", local_to_global_mat)

        current_time = time.time()
        if current_time - self.start_time >= 5:
            self.save_data_with_pickle()
            self.start_time = current_time


    def update_points(self, points):
        self.current_points = points
        self._draw_scene(points)

    def start_polling_data(self):
        self._poll_data()

    
    def _poll_data(self):
        try:
            while not data_queue.empty():
                item = data_queue.get_nowait()
                if item[0] == 'mocap':
                    frameNumber, points = item[1], item[2]
                    current_time = time.time()
                    if current_time - self.last_mocap_print_time >= self.mocap_print_interval:
                        self.last_mocap_print_time = current_time
                        # print(f"\n[MoCap 帧] Frame #{frameNumber}")
                        # for i, pos in enumerate(points):
                        #     print(f"\tMarker {i}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
                    self.update_points(points)

                elif item[0] == 'serial':
                    values = item[1]  # 4个数
                    current_time = time.time()
                    if current_time - self.last_serial_print_time >= self.serial_print_interval:
                        self.last_serial_print_time = current_time
                        # values[2] = values[2] / 3830
                        # values[1] = values[1] / 2110
                        # values[0] = values[0] / 1800
                        # values[3] = values[3] / 3810
                        
                        f1 = values[2] - values[0] 
                        f2 = values[1] - values[3] 
                        f3 = -sum(values)
                        self.latest_force = [f1, f2, f3]
                        self.latest_raw_data = values
                        # print(f"[串口力] 力1={f1:.3f}, 力2={f2:.3f}, 力3={f3:.3f}")
                
                elif item[0] == 'ati_force':
                    fx, fy, fz, mx, my, mz = item[1]
                    current_time = time.time()
                    if current_time - self.last_serial_truth_force_print_time >= self.serial_truth_force_print_interval:
                        self.last_serial_truth_force_print_time = current_time
                        self.latest_truth_force = [fx, fy, fz, mx, my, mz]

        except queue.Empty:
            pass
        self.master.after(30, self._poll_data)

    def clear_path(self):
        self.path_points = []
        if self.current_points is not None:
            self._draw_scene(self.current_points, record_tip=False)
    
    def save_data_with_pickle(self, filename=save_name):
        # 创建一个包含两个数据的字典
        data_to_save = {
            'truth_forces': self.truth_forces,
            'path_forces': self.path_forces
        }
        
        # 使用pickle保存数据
        with open(filename, 'wb') as f:  # 注意使用二进制写入模式 'wb'
            pickle.dump(data_to_save, f)


import serial
import serial.tools.list_ports
def start_serial_listener():
    try:
        ser = serial.Serial(
            port='COM13',
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05
        )
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return
    
    buffer = bytearray()
    expecting_header = True

    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                buffer.extend(data)

                while len(buffer) >= 2:
                    if expecting_header:
                        # 查找帧头 0xAA 0x55
                        if buffer[0] == 0xAA and buffer[1] == 0x55:
                            expecting_header = False
                            del buffer[:2]
                        else:
                            del buffer[0]
                    else:
                        if len(buffer) >= 14:
                            # 提取数据段（12字节）+ 校验帧尾（2字节）
                            payload = buffer[:12]
                            tail1 = buffer[12]
                            tail2 = buffer[13]

                            if tail1 == 0x55 and tail2 == 0xAA:
                                # 解析前4个通道（共8字节）
                                values = []
                                for i in range(0, 8, 2):
                                    val = (payload[i] << 8) | payload[i + 1]
                                    values.append(val)

                                # 放入队列
                                data_queue.put(('serial', values))
                            else:
                                print("[串口] 帧尾错误，跳过该帧")

                            del buffer[:14]
                            expecting_header = True
                        else:
                            break  # 等待更多数据

    except Exception as e:
        print(f"[串口] 监听错误: {e}")
    finally:
        ser.close()
        print("[串口] 已关闭")


def start_natnet():

    ip = "127.0.0.1"

    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(ip)

    try:
        while True:
            # 0:阻塞接收 1：非阻塞接收
            frame = LuMoSDKClient.ReceiveData(0)
            if frame is None:
                continue
                
            # 获取帧号
            frameNumber = frame.FrameId
            # 获取标记点
            markers = frame.markers
            
            # 坐标转换
            transformed_Pos = []
            for marker in markers:
                transformed_x = marker.X / 1000
                transformed_y = -marker.Z / 1000
                transformed_z = marker.Y / 1000
                transformed_Pos.append([transformed_x, transformed_y, transformed_z])

            # 保持与之前相同的逻辑：如果有至少6个点，取前6个放入队列
            if len(transformed_Pos) >= 6:
                data_queue.put(('mocap', frameNumber, transformed_Pos[:6]))
                
    except KeyboardInterrupt:
        # 处理 Ctrl+C 退出
        print("Stopping NatNet client...")
    finally:
        # # 断开连接，释放资源
        # LuMoSDKClient.Disconnect()  # 假设存在这个方法
        print("Disconnected")



import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TaskMode
readtimes = 0
key = '0'
FT39767 = np.array([[0.01712, -0.02956, 0.09076, -1.66120, -0.04421, 1.65367],
                    [- 0.06833, 2.07196, 0.06375, -0.95731, 0.03652, -0.91896],
                    [1.86349, -0.05178, 1.90941, 0.00516, 1.82368, 0.00703],
                    [- 0.33522, 12.58865, 10.95500, -5.84491, -9.98641, -5.61671],
                    [- 12.24102, 0.50841, 5.89063, 10.17926, 6.15407, -10.11660],
                    [- 0.20864, 7.88837, -0.45651, 7.27192, -0.18374, 7.20123]])
def ati_read():
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5")
        task.timing.cfg_samp_clk_timing(rate=1000, sample_mode=AcquisitionType.CONTINUOUS)
        f = open('ATLsensor1025.txt', 'a+')
        global readtimes
        global key

        data = task.read(number_of_samples_per_channel=10)
        data_array = np.array(data)
        data_avg = np.mean(data_array, axis=1)
        base = np.dot(FT39767, data_avg)
        while 1:
            while key=='0':  # 采集数据
                try:
                    readtimes = readtimes + 1
                    data = task.read(number_of_samples_per_channel=10)
                    data_array = np.array(data)
                    data_avg = np.mean(data_array, axis=1)
                    Force = np.dot(FT39767, data_avg)-base
                    fx, fy, fz, mx, my, mz = Force
                    data_queue.put(('ati_force', (fx, fy, fz, mx, my, mz)))
                    # print('Output vealue: %d %f %f %f %f %f %f' % (
                    # readtimes, OutputFPGA_1, OutputFPGA_2, OutputFPGA_3, OutputFPGA_4, OutputFPGA_5, OutputFPGA_6))
                    # print(readtimes, OutputFPGA_1, OutputFPGA_2, OutputFPGA_3, OutputFPGA_4, OutputFPGA_5, OutputFPGA_6, file=f)

                except KeyboardInterrupt:
                    f.close()
                    break
      
      


# ---------------------------- main ----------------------------

if __name__ == '__main__':

        # 启动串口监听线程
    serial_thread = threading.Thread(target=start_serial_listener, daemon=True)
    serial_thread.start()

    mocap_thread = threading.Thread(target=start_natnet, daemon=True)
    mocap_thread.start()

    ati_thread = threading.Thread(target=ati_read, daemon=True)
    ati_thread.start()


    root = tk.Tk()
    gui = BrushGUI(root)
    gui.start_polling_data()
    root.mainloop()
