import sys
import serial
from serial.tools import list_ports  # 用于检测可用串口
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
from PyQt5.QtCore import QSettings

class SerialPlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口
        self.setWindowTitle("ADC数据可视化")
        self.resize(1200, 800)
        
        # 创建主控件和布局
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # 创建串口设置区域
        self.setup_serial_ui(layout)
        
        # 创建标签显示当前值
        self.value_labels = {}
        value_layout = QtWidgets.QHBoxLayout()
        
        # 创建网格布局来组织标签
        grid_layout = QtWidgets.QGridLayout()
        
        # 添加通道标签
        for i in range(1, 5):
            label = QtWidgets.QLabel(f"通道{i}:")
            adc_label = QtWidgets.QLabel("0")
            voltage_label = QtWidgets.QLabel("0.000V")
            
            # 设置标签样式使其更清晰
            label.setStyleSheet("font-weight: bold;")
            adc_label.setStyleSheet("min-width: 60px;")
            voltage_label.setStyleSheet("min-width: 70px; color: blue;")
            
            grid_layout.addWidget(label, 0, (i-1)*2)
            grid_layout.addWidget(adc_label, 0, (i-1)*2+1)
            grid_layout.addWidget(voltage_label, 1, (i-1)*2+1)
            
            self.value_labels[f'result0{i}_adc'] = adc_label
            self.value_labels[f'result0{i}_voltage'] = voltage_label
        
        # 添加ADV和V12标签
        adv_label = QtWidgets.QLabel("ADV:")
        adv_adc = QtWidgets.QLabel("0")
        adv_voltage = QtWidgets.QLabel("0.000V")
        
        v12_label = QtWidgets.QLabel("V12:")
        v12_adc = QtWidgets.QLabel("0")
        v12_voltage = QtWidgets.QLabel("0.000V")
        
        # 设置样式
        adv_label.setStyleSheet("font-weight: bold;")
        v12_label.setStyleSheet("font-weight: bold;")
        adv_adc.setStyleSheet("min-width: 60px;")
        adv_voltage.setStyleSheet("min-width: 70px; color: blue;")
        v12_adc.setStyleSheet("min-width: 60px;")
        v12_voltage.setStyleSheet("min-width: 70px; color: blue;")
        
        grid_layout.addWidget(adv_label, 2, 0)
        grid_layout.addWidget(adv_adc, 2, 1)
        grid_layout.addWidget(adv_voltage, 2, 2)
        
        grid_layout.addWidget(v12_label, 2, 3)
        grid_layout.addWidget(v12_adc, 2, 4)
        grid_layout.addWidget(v12_voltage, 2, 5)
        
        self.value_labels['resultadv_adc'] = adv_adc
        self.value_labels['resultadv_voltage'] = adv_voltage
        self.value_labels['resultv12_adc'] = v12_adc
        self.value_labels['resultv12_voltage'] = v12_voltage
        
        # 添加ADC参考电压信息
        ref_label = QtWidgets.QLabel("ADC参考电压: 3.3V (12位分辨率)")
        ref_label.setStyleSheet("color: gray; font-size: 10px;")
        grid_layout.addWidget(ref_label, 3, 0, 1, 6, QtCore.Qt.AlignCenter)
        
        value_layout.addLayout(grid_layout)
        layout.addLayout(value_layout)
        
        # 创建标签页
        self.tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 创建第一个图表 - 4条曲线（电压值）
        self.plot1 = pg.PlotWidget(title="ADC通道1-4 (电压值)")
        self.plot1.addLegend()
        self.plot1.setLabel('left', '电压 (V)')
        self.plot1.setLabel('bottom', '时间')
        self.plot1.showGrid(x=True, y=True)
        
        # 创建第二个图表 - 2条曲线（电压值）
        self.plot2 = pg.PlotWidget(title="ADV和V12 (电压值)")
        self.plot2.addLegend()
        self.plot2.setLabel('left', '电压 (V)')
        self.plot2.setLabel('bottom', '时间')
        self.plot2.showGrid(x=True, y=True)
        
        # 添加到标签页
        self.tab_widget.addTab(self.plot1, "主通道")
        self.tab_widget.addTab(self.plot2, "辅助通道")
        
        # 初始化数据存储
        self.max_data_points = 500
        self.data = {
            'result01': deque(maxlen=self.max_data_points),
            'result02': deque(maxlen=self.max_data_points),
            'result03': deque(maxlen=self.max_data_points),
            'result04': deque(maxlen=self.max_data_points),
            'resultadv': deque(maxlen=self.max_data_points),
            'resultv12': deque(maxlen=self.max_data_points),
            'time': deque(maxlen=self.max_data_points)
        }
        
        # 初始化曲线
        self.curves = {
            'result01': self.plot1.plot(pen='r', name='通道1'),
            'result02': self.plot1.plot(pen='g', name='通道2'),
            'result03': self.plot1.plot(pen='b', name='通道3'),
            'result04': self.plot1.plot(pen='y', name='通道4'),
            'resultadv': self.plot2.plot(pen='c', name='ADV'),
            'resultv12': self.plot2.plot(pen='m', name='V12')
        }
        
        # 串口对象
        self.serial_port = None
        self.serial_timer = QtCore.QTimer()
        self.serial_timer.timeout.connect(self.read_serial_data)
        
        # 数据解析状态
        self.buffer = bytearray()
        self.expecting_header = True
        self.data_packet = []
        
        # ADC参数
        self.adc_reference_voltage = 3.3  # 参考电压
        self.adc_resolution = 4095        # 12位ADC的最大值 (2^12 - 1)
        
        # 加载上次的串口设置
        self.load_serial_settings()
    
    def setup_serial_ui(self, layout):
        serial_layout = QtWidgets.QHBoxLayout()
        
        # 串口选择
        port_label = QtWidgets.QLabel("串口:")
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setMinimumWidth(150)
        serial_layout.addWidget(port_label)
        serial_layout.addWidget(self.port_combo)
        
        # 波特率选择
        baud_label = QtWidgets.QLabel("波特率:")
        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baud_combo.setCurrentText("115200")  # 默认波特率设为115200
        serial_layout.addWidget(baud_label)
        serial_layout.addWidget(self.baud_combo)
        
        # 连接/断开按钮
        self.connect_btn = QtWidgets.QPushButton("连接")
        self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.connect_btn.clicked.connect(self.toggle_serial_connection)
        serial_layout.addWidget(self.connect_btn)
        
        # 刷新串口按钮
        refresh_btn = QtWidgets.QPushButton("刷新串口")
        refresh_btn.clicked.connect(self.refresh_serial_ports)
        serial_layout.addWidget(refresh_btn)
        
        layout.addLayout(serial_layout)
    
    def refresh_serial_ports(self):
        """刷新可用串口列表"""
        self.port_combo.clear()
        ports = list_ports.comports()
        if ports:
            for port in ports:
                self.port_combo.addItem(port.device)
        else:
            self.port_combo.addItem("未检测到串口")
    
    def load_serial_settings(self):
        """加载上次使用的串口设置"""
        settings = QSettings("MyCompany", "SerialPlotter")
        last_port = settings.value("last_port", "")
        last_baud = settings.value("last_baud", "115200")
        
        # 设置波特率
        if last_baud in ["9600", "19200", "38400", "57600", "115200"]:
            self.baud_combo.setCurrentText(last_baud)
        
        # 刷新串口列表
        self.refresh_serial_ports()
        
        # 设置串口
        if last_port:
            index = self.port_combo.findText(last_port)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)
            else:
                # 如果上次的串口不存在，添加到列表中
                self.port_combo.addItem(last_port)
                self.port_combo.setCurrentIndex(self.port_combo.count() - 1)
        elif self.port_combo.count() > 0:
            # 如果没有保存的设置，选择第一个可用的串口
            self.port_combo.setCurrentIndex(0)
    
    def save_serial_settings(self):
        """保存当前串口设置"""
        settings = QSettings("MyCompany", "SerialPlotter")
        settings.setValue("last_port", self.port_combo.currentText())
        settings.setValue("last_baud", self.baud_combo.currentText())
    
    def toggle_serial_connection(self):
        if self.serial_port and self.serial_port.is_open:
            self.disconnect_serial()
            self.connect_btn.setText("连接")
            self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        else:
            if self.connect_serial():
                self.connect_btn.setText("断开")
                self.connect_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
    
    def connect_serial(self):
        port = self.port_combo.currentText()
        baudrate = int(self.baud_combo.currentText())
        
        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=0.1)
            self.serial_timer.start(10)  # 每10ms检查一次串口数据
            print(f"已连接到 {port} @ {baudrate} baud")
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"无法打开串口:\n{str(e)}")
            return False
    
    def disconnect_serial(self):
        if self.serial_port:
            self.serial_timer.stop()
            self.serial_port.close()
            self.serial_port = None
            print("串口已断开")
    
    def read_serial_data(self):
        if not self.serial_port or not self.serial_port.is_open:
            return
        
        try:
            data = self.serial_port.read(self.serial_port.in_waiting or 1)
            if data:
                self.buffer.extend(data)
                self.process_buffer()
        except Exception as e:
            print(f"读取串口数据错误: {str(e)}")
            self.disconnect_serial()
            self.connect_btn.setText("连接")
            self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
    
    def process_buffer(self):
        while len(self.buffer) >= 2:
            if self.expecting_header:
                # 查找AA 55头
                if self.buffer[0] == 0xAA and self.buffer[1] == 0x55:
                    self.expecting_header = False
                    self.data_packet = []
                    del self.buffer[:2]
                else:
                    del self.buffer[0]
            else:
                # 需要读取14个字节 (6个通道 × 2字节) + 2字节的结束标志
                if len(self.buffer) >= 14:
                    # 提取6个通道的数据 (每个通道2字节)
                    channels = []
                    for i in range(0, 14, 2):
                        if i+1 < len(self.buffer):
                            value = (self.buffer[i] << 8) | self.buffer[i+1]
                            channels.append(value)
                    
                    # 检查结束标志 (55 AA)
                    if self.buffer[12] == 0x55 and self.buffer[13] == 0xAA:
                        # 更新数据
                        self.update_data(channels[:6])
                    
                    # 无论是否成功解析，都删除这14个字节
                    del self.buffer[:14]
                    self.expecting_header = True
                else:
                    break
    
    def adc_to_voltage(self, adc_value):
        """将ADC值转换为电压值"""
        return (adc_value / self.adc_resolution) * self.adc_reference_voltage
    
    def update_data(self, values):
        if len(values) != 6:
            return
        
        # 获取当前时间戳
        current_time = len(self.data['time'])
        if self.data['time']:
            current_time = self.data['time'][-1] + 1
        
        # 更新数据
        self.data['time'].append(current_time)
        
        channel_names = ['result01', 'result02', 'result03', 'result04', 'resultadv', 'resultv12']
        for name, value in zip(channel_names, values):
            # 存储原始ADC值
            self.data[name].append(value)
            
            # 计算电压值
            voltage = self.adc_to_voltage(value)
            
            # 更新标签显示 - ADC值和电压值
            self.value_labels[f'{name}_adc'].setText(f"{value}")
            self.value_labels[f'{name}_voltage'].setText(f"{voltage:.3f}V")
        
        # 更新曲线
        self.update_plots()
    
    def update_plots(self):
        time_data = list(self.data['time'])
        
        # 更新第一个图表 (通道1-4) - 使用电压值
        self.curves['result01'].setData(time_data, [self.adc_to_voltage(x) for x in self.data['result01']])
        self.curves['result02'].setData(time_data, [self.adc_to_voltage(x) for x in self.data['result02']])
        self.curves['result03'].setData(time_data, [self.adc_to_voltage(x) for x in self.data['result03']])
        self.curves['result04'].setData(time_data, [self.adc_to_voltage(x) for x in self.data['result04']])
        
        # 更新第二个图表 (ADV和V12) - 使用电压值
        self.curves['resultadv'].setData(time_data, [self.adc_to_voltage(x) for x in self.data['resultadv']])
        self.curves['resultv12'].setData(time_data, [self.adc_to_voltage(x) for x in self.data['resultv12']])
    
    def closeEvent(self, event):
        # 保存当前串口设置
        self.save_serial_settings()
        self.disconnect_serial()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SerialPlotter()
    window.show()
    sys.exit(app.exec_())