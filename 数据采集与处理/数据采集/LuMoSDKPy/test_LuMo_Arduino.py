import threading
import time
import serial
import LuMoSDKClient as LuMoSDKClient
import os

latest_motion_data = None
latest_arduino_data = None
data_lock = threading.Lock()
stop_event = threading.Event()

print("保存目录:", os.getcwd())  # 打印当前保存目录

def motion_capture_thread():
    global latest_motion_data
    ip = "127.0.0.1"
    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(ip)

    while not stop_event.is_set():
        frame = LuMoSDKClient.ReceiveData(0)
        if frame is None:
            continue
        markers_info = [(marker.Id, marker.X, marker.Y, marker.Z) for marker in frame.markers]
        with data_lock:
            latest_motion_data = markers_info

def arduino_thread():
    global latest_arduino_data
    ser = serial.Serial('COM7', 115200)
    time.sleep(2)
    while not stop_event.is_set():
        line = ser.readline()
        try:
            data = [int(i) for i in line.decode('utf-8').strip().split()]
        except:
            continue
        with data_lock:
            latest_arduino_data = data

def data_write_thread():
    with open("sync_data.txt", "w", buffering=1) as f:  # line-buffered，写一行就落盘
        while not stop_event.is_set() or (latest_motion_data and latest_arduino_data):
            time.sleep(0.01)
            with data_lock:
                if latest_motion_data and latest_arduino_data:
                    marker_str_list = []
                    for marker in latest_motion_data:
                        marker_str_list.extend([str(marker[0]), str(marker[1]), str(marker[2]), str(marker[3])])
                    arduino_str_list = [str(v) for v in latest_arduino_data]
                    line = ",".join(marker_str_list + arduino_str_list) + "\n"
                    f.write(line)
                    print("Motion markers:")
                    for marker in latest_motion_data:
                        print(f"ID: {marker[0]}, X: {marker[1]}, Y: {marker[2]}, Z: {marker[3]}")
                    print("Arduino values:", latest_arduino_data)
                    print("="*50)
                    latest_motion_data.clear()
                    latest_arduino_data.clear()

if __name__ == "__main__":
    t1 = threading.Thread(target=motion_capture_thread)
    t2 = threading.Thread(target=arduino_thread)
    t3 = threading.Thread(target=data_write_thread)

    t1.start()
    t2.start()
    t3.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("停止采集中，请稍候...")
        stop_event.set()
        t1.join()
        t2.join()
        t3.join()  # 等写线程完整结束
        print("采集已停止，文件保存完成。")




