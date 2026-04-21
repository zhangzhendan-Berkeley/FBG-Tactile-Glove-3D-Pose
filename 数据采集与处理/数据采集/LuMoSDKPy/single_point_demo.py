import sys
import time
import serial
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

class SerialThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QThread.__init__(self)
        self.serial_port = serial.Serial('COM6', 115200)  # Change this to your serial port!

    def __del__(self):
        self.wait()

    def run(self):
        while True:
            data = self.serial_port.readline()
            data = data.decode('utf-8').strip().split()
            data = [int(i) for i in data]
            self.signal.emit(data)

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')
        self.font = QFont("Calibri", 14)
        self.graphWidget.setFont(self.font)
        self.graphWidget.setTitle("Intensity of the Clear Channel", size="15pt")
        self.graphWidget.setLabel('left', "Light Intensity", size="10pt")
        self.graphWidget.setLabel('bottom', "Time (s)", size="10pt")

        self.color_label = QLabel()

        self.layout.addWidget(self.graphWidget)
        self.layout.addWidget(self.color_label)
        self.color_label.setFont(self.font)
        self.color_label.setText("Colors received by RGB sensor")
        self.color_label.setMinimumHeight(200)  # adjust as needed

        self.x = list(np.linspace(0, 10, 100))  # 100 time points between 0 and 10
        self.y = [0 for _ in range(100)]  # 100 data points

        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pg.mkPen('k', width=3))

        self.serial_thread = SerialThread()
        self.serial_thread.signal.connect(self.update_plot_data)
        self.serial_thread.start()
        self.setLayout(self.layout)

    def update_plot_data(self, data):
        self.y.append(data[3])  # Add the latest y value.
        self.y = self.y[-200:]  # Keep only the last 100 y values.

        self.x.append(self.x[-1] + 0.025)  # Add a new x value. Assuming data is received every 0.1 seconds.
        self.x = self.x[-200:]  # Keep only the last 100 x values.

        self.data_line.setData(self.x, self.y)  # Update the data.

        # Update the RGB color
        normalized_color = self.normalize_to_rgb(data[:3])
        self.color_label.setStyleSheet(f"background-color: rgb{tuple(normalized_color)};")

        # Invalidate and activate layout to ensure proper resizing
        self.layout.invalidate()
        self.layout.activate()

    def normalize_to_rgb(self, values):
        max_val = max(values)
        if max_val == 0:
            return [0, 0, 0]
        factor = 255 / max_val
        return [int(value * factor) for value in values]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Calibri", 14)  # Set the font to Calibri and size to 10
    app.setFont(font)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
