#!/usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QFileDialog
from PyQt5 import uic
import json
import sys
import numpy as np
import pyqtgraph as pg
from scipy import optimize
import os
DIR = os.getcwd() + '/'
DIR_DATA = os.getcwd() + '/data/'


class SCProc(QMainWindow):
    def __init__(self):
        super(SCProc, self).__init__()
        uic.loadUi("sc_plot.ui", self)
        self.EXPERIMENT_TIME = 30  # ns - set by user
        self.EXPERIMENT_CALIBRATION = 0  # ns per dot
        self.x = np.arange(0, 1280, 1)
        self.load_flag = 0
        self.files_list = os.listdir(DIR_DATA)

        self.profile_data = {}
        self.image_data = {}
        self.fft_data = {}
        self.aux_data = {}

        self.show()
        self.plot_area()
        self.calibrate()

        self.btn_start.clicked.connect(self.main_loop)
        self.spin_sample.valueChanged.connect(self.sample_replot)
        self.btn_save.clicked.connect(self.data_save)
        self.btn_load.clicked.connect(self.data_load)

    def main_loop(self):
        self.load_flag = 0
        num = 0
        # sorting by date
        self.files_list = sorted(os.listdir(DIR_DATA), key=lambda x: os.path.getctime(os.path.join(DIR_DATA, x)))
        print(self.files_list)
        self.files_list = ['beam_00000030.pgm']
        for file in self.files_list:
            num += 1
            try:
                self.aux_data[str(num)] = file.split('_')[2]
            except IndexError:
                self.aux_data[str(num)] = 0
            data = np.transpose(np.loadtxt(DIR_DATA + file, skiprows=4))
            self.data_proc(data, str(num))

        # saving profiles for modelating
        # a = np.array(self.profile_data['1'])
        # for i in range(2, 5):
        #     a = np.vstack((a, self.profile_data[str(i)]))
        # np.savetxt('exclusive_profs.txt', a)

        self.slider_sample.setRange(1, len(self.files_list))
        self.spin_sample.setRange(1, len(self.files_list))
        self.sample_replot()

    def calibrate(self):
        x_data = np.empty(0)
        y_data = np.empty(0)
        calib = np.transpose(np.loadtxt(DIR + 'calib.pgm', skiprows=4))

        for row in range(50, 450, 25):
            calib_sliced = calib[:, row]
            val_up = np.where(calib_sliced > calib_sliced.max() * 4 / 5)
            x_data = np.append(x_data, val_up[0][0])
            y_data = np.append(y_data, row)
            x_data = np.append(x_data, val_up[0][-1])
            y_data = np.append(y_data, row)
        self.profile_plot.clear()
        self.profile_plot.plot(x_data, y_data, pen=None, symbol='o')
        try:
            circlefit = lambda p, x: - np.sqrt(p[0] ** 2 - (x - p[1]) ** 2) + p[2]
            errfunc = lambda p, x, y: circlefit(p, x) - y_data
            p = [700, np.mean(x_data), 450]
            p_fit, success = optimize.leastsq(errfunc, p[:], args=(x_data, y_data))
            print(p_fit)
            self.statusbar.showMessage('calibration is done, R = %0.2f pixels' % (p_fit[0]))
            self.EXPERIMENT_CALIBRATION = self.EXPERIMENT_TIME / 2 / p_fit[0]
            self.beam_plot.setImage(calib)
            self.profile_plot.plot(np.arange(0, 1300, 1), circlefit(p_fit, np.arange(0, 1300, 1)), pen=None,
                                   symbol='star', symbolSize=5)
        except RuntimeWarning as exc:
            print(exc)

    def data_proc(self, data, num):
        self.image_data[num] = data
        profile = np.sum(data, axis=0)

        gaussfit = lambda p, x: p[0] * np.exp(-(((x - p[1]) / p[2]) ** 2) / 2) + p[3]
        errfunc = lambda p, x, y: gaussfit(p, x) - profile
        p = [4e5, profile.argmax(), 20, 0]
        p_fit, success = optimize.leastsq(errfunc, p[:], args=(np.arange(len(profile)), profile))
        print(p_fit, num)
        background = np.mean(data[400:600, 200:200+int(6*p_fit[2])], axis=1)
        sliced_profile = np.mean(data[:, int(p_fit[1] - 3*p_fit[2]):int(p_fit[1] + 3*p_fit[2])], axis=1)  # - np.mean(background)
        fft = np.fft.rfft((sliced_profile - np.mean(sliced_profile)), len(sliced_profile))
        freq = np.fft.rfftfreq(len(sliced_profile), self.EXPERIMENT_CALIBRATION * 1e-9)

        # beam = np.where(sliced_profile > 0)
        # for_save = sliced_profile[beam[0][0]:beam[0][-1]] / max(sliced_profile)
        # np.savetxt('buffer.txt', np.vstack((for_save, for_save)))

        self.profile_data[num] = sliced_profile  # / max(sliced_profile)
        self.fft_data[num] = (freq, np.sqrt(fft.real**2 + fft.imag**2))
        self.progress_bar.setValue(int(num) / len(self.files_list) * 100)

    def sample_replot(self):
        print(self.EXPERIMENT_CALIBRATION)
        sample = str(self.spin_sample.value())
        profile_data = self.profile_data[sample]
        x = self.x * self.EXPERIMENT_CALIBRATION / 3

        if not self.load_flag:
            profile_data = self.profile_data[sample]
            image = self.image_data[sample]
            freq, fft = self.fft_data[sample]

            self.profile_plot.clear()
            self.fft_plot.clear()

            self.beam_plot.setImage(image)
            self.profile_plot.plot(x, profile_data, pen=pg.mkPen('r', width=3))
            self.fft_plot.plot(freq / 1e9, fft, pen=pg.mkPen('g', width=3))

        self.profile_plot.clear()
        self.profile_plot.plot(x, profile_data, pen=pg.mkPen('r', width=3))

    def data_save(self):
        # np.savetxt(str(self.aux_data[self.spin_sample.value()]) + '.txt', self.profile_data[self.spin_sample.value()])
        # self.statusbar.showMessage('data N %d saved' % (self.spin_sample.value()))

        save_dir = QFileDialog.getSaveFileName(parent=self, directory=DIR + 'saved_data', filter='Text Files (*.txt)')
        if save_dir:
            file_name = save_dir[0]
            save_file = open(file_name, 'w')
            save_file.write(str(self.EXPERIMENT_CALIBRATION))
            save_file.write('\n')
            s_profile_data = {key: np.ndarray.tolist(val) for key, val in self.profile_data.items()}
            save_file.write(json.dumps(s_profile_data))
            save_file.close()
            self.statusbar.showMessage('data saved')

    def data_load(self):
        self.load_flag = 1
        load_file = QFileDialog.getOpenFileName(parent=self, directory=DIR + 'saved_data', filter='Text Files (*.txt)')
        file_name = load_file[0]
        load_file_ = open(file_name, 'r')
        load_data = load_file_.readlines()
        self.EXPERIMENT_CALIBRATION = float(load_data[0])
        self.profile_data = json.loads(load_data[1])

        self.slider_sample.setRange(1, len(self.profile_data))
        self.spin_sample.setRange(1, len(self.profile_data))
        self.sample_replot()

    def plot_area(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)

        # beam image
        self.beam_plot = pg.ImageView(parent=self)
        self.beam_plot.ui.histogram.hide()
        self.beam_plot.ui.roiBtn.hide()
        self.beam_plot.ui.menuBtn.hide()

        self.plot_win = pg.GraphicsWindow(parent=self)
        # beam profile
        self.profile_plot = self.plot_win.addPlot(enableMenu=False)
        self.profile_plot.showGrid(x=True, y=True)
        self.profile_plot.setLabel('left', "Ampl", units='a.u.')
        self.profile_plot.setLabel('bottom', "Z", units='m')

        self.plot_win.nextRow()
        # beam fft
        self.fft_plot = self.plot_win.addPlot(enableMenu=False)
        self.fft_plot.showGrid(x=True, y=True)
        self.fft_plot.setLabel('left', "Ampl", units='a.u.')
        self.fft_plot.setLabel('bottom', "F", units='GHz')

        p_beam = QGridLayout()
        self.output_beam.setLayout(p_beam)
        p_beam.addWidget(self.beam_plot)

        p_plot = QGridLayout()
        self.output_plot.setLayout(p_plot)
        p_plot.addWidget(self.plot_win)


if __name__ == "__main__":
    app = QApplication(['mv'])
    w = SCProc()
    sys.exit(app.exec_())
