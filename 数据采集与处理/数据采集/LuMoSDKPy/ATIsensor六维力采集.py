import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TaskMode
import threading

global readtimes
global key

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
                    OutputFPGA_1 = Force[0]
                    OutputFPGA_2 = Force[1]
                    OutputFPGA_3 = Force[2]
                    OutputFPGA_4 = Force[3]
                    OutputFPGA_5 = Force[4]
                    OutputFPGA_6 = Force[5]
                    print('Output vealue: %d %.4f %.4f %.4f %.4f %.4f %.4f' % (
                    readtimes, OutputFPGA_1, OutputFPGA_2, OutputFPGA_3, OutputFPGA_4, OutputFPGA_5, OutputFPGA_6))
                    print(readtimes, OutputFPGA_1, OutputFPGA_2, OutputFPGA_3, OutputFPGA_4, OutputFPGA_5, OutputFPGA_6, file=f)

                except KeyboardInterrupt:
                    f.close()
                    break

            if key == 'c':  # calibration
                print('calibration')
                data = task.read(number_of_samples_per_channel=10)
                data_array = np.array(data)
                data_avg = np.mean(data_array, axis=1)
                base = np.dot(FT39767, data_avg)
                key = '0'

            elif key == 'q':  # quit
                print('quit')
                f.close()
                break

            else:
                key='0'


def keyboard():
    global key
    while 1:
        key = input()
        print(key)
        if key == 'q':
            break

def main():
    print('请输入key值：0——开始数据采集 c——校准传感器，q——退出程序，其他按键默认为0')
    global key
    key ='0'
    key=input()
    threading.Thread(target=ati_read).start()
    threading.Thread(target=keyboard).start()


if __name__ == '__main__':
    main()
