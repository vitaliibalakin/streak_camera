#!/usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout
from PyQt5 import uic
import sys
import numpy as np
import pyqtgraph as pg
from scipy import optimize
import os
DIR_CALIB = os.getcwd() + '/'
DIR_DATA = os.getcwd() + '/data/'


class SCProc(QMainWindow):
    def __init__(self):
        super(SCProc, self).__init__()
        uic.loadUi("sc_plot.ui", self)
        self.show()
        self.plot_area()
        self.calibrate()

        # for file in os.listdir(DIR_DATA):
        #     data = np.transpose(np.loadtxt(DIR_DATA + file, skiprows=4))
        #     self.data_proc(data)

    def calibrate(self):
        x_data = np.ndarray([])
        y_data = np.ndarray([])
        calib = np.transpose(np.loadtxt(DIR_CALIB + 'calib.pgm', skiprows=4))
        self.profile_plot.plot(calib[:, 50], pen=pg.mkPen('r', width=1))
        self.beam_plot.setImage(calib)
        for row in range(50, 250, 50):
            calib_sliced = calib[:, row]
            val_up = np.where(calib_sliced > calib_sliced.max() * 4 / 5)
            x_data = np.append(x_data, val_up[0][0])
            y_data = np.append(y_data, row)
            x_data = np.append(x_data, val_up[0][-1])
            y_data = np.append(y_data, row)
        self.profile_plot.clear()
        self.profile_plot.plot(x_data, y_data, pen=None, symbol='o')

        circlefit = lambda p, x: - np.sqrt(p[0] ** 2 - (x - p[1]) ** 2) + p[2]
        errfunc = lambda p, x, y: circlefit(p, x) - y_data
        p = [650, np.mean(x_data), 500]
        p_fit, success = optimize.leastsq(errfunc, p[:], args=(x_data, y_data))
        print(p_fit)

    def data_proc(self, data):
        profile = np.sum(data, axis=0)

        gaussfit = lambda p, x: p[0] * np.exp(-(((x - p[1]) / p[2]) ** 2) / 2) + p[3]
        errfunc = lambda p, x, y: gaussfit(p, x) - profile
        p = [4e5, profile.argmax(), 20, 0]
        p_fit, success = optimize.leastsq(errfunc, p[:], args=(np.arange(len(profile)), profile))
        sliced_data = data[:, int(p_fit[1] - 3*p_fit[2]):int(p_fit[1] + 3*p_fit[2])]

        self.beam_plot.setImage(sliced_data)
        self.profile_plot.clear()
        self.profile_plot.plot(np.sum(sliced_data, axis=1), pen=pg.mkPen('r', width=1))

    def plot_area(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # beam image
        self.beam_plot = pg.ImageView(parent=self)
        self.beam_plot.ui.histogram.hide()
        self.beam_plot.ui.roiBtn.hide()
        self.beam_plot.ui.menuBtn.hide()

        # beam profile
        self.plot_win = pg.GraphicsWindow(parent=self)
        self.profile_plot = self.plot_win.addPlot(enableMenu=False)
        self.profile_plot.showGrid(x=True, y=True)
        self.profile_plot.setLabel('left', "Ampl", units='a.u.')
        self.profile_plot.setLabel('bottom', "Time", units='ns')

        p_beam = QGridLayout()
        self.output_beam.setLayout(p_beam)
        p_beam.addWidget(self.beam_plot)

        p_profile = QGridLayout()
        self.output_profile.setLayout(p_profile)
        p_profile.addWidget(self.plot_win)


if __name__ == "__main__":
    app = QApplication(['mv'])
    w = SCProc()
    sys.exit(app.exec_())
