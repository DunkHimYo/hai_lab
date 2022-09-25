from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer,Qt
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np


class Main(QtWidgets.QMainWindow):

    def __init__(self, widget_stack):
        super().__init__()
        uic.loadUi('./veem_gui/ui/main.ui', self)

        self.widget=widget_stack
        print(self.widget.count())
        print(widget_stack.currentIndex())
        self._chk_data = {'vr': False, 'ep': False, 'eeg': False, 'mri': False}

        self.action_vr.triggered.connect(self._import_vr_data)
        self.action_ep.triggered.connect(self._import_ep_data)
        self.action_eeg.triggered.connect(self._import_eeg_data)
        self.action_mri.triggered.connect(self._import_mri_data)


        self.vr_title.mousePressEvent=self._init_next_page
        #self.action_export.triggered.connect(lambda: self._init_pop_up(DataSetExportPopUp))

        """
        self._show_vr_features()
        self._show_ep_features()
        self._show_eeg_features()
        self._show_mri_features()
        self._show_total_result()

        self._show_vr_result()
        self._show_ep_result()
        self._show_eeg_result()
        self._show_mri_result()
        """

    def _attribute(self):
        self.widget_stack = widget_stack
        self.data_set_path=None
        self.raw_eeg=None
        self.ch_info={}

    def _init_pop_up(self,cls):
        popup_page = cls(self)
        popup_page.exec()

    def _init_next_page(self,event):
        current_page=DetailAboutVEEM(self)
        self.widget.addWidget(current_page)
        self.widget.setCurrentWidget(current_page)

    def _import_data_path(self):
        file_name, selected_filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open file','./",
                                                                                     "csv(*.csv)")
        return file_name

    def _import_vr_data(self):
        file_name = self._import_data_path()

        if file_name:
            df = pd.read_csv(file_name, index_col=0)
            self.create_table_widget(self.vr_table, df)
            self._chk_data['vr']=True

    def _import_ep_data(self):
        file_name=self._import_data_path()
        if file_name:
            df = pd.read_csv(file_name, index_col=0)
            self.create_table_widget(self.ep_table, df)
            self._chk_data['ep'] = True

    def _import_eeg_data(self):
        file_name = self._import_data_path()
        if file_name:
            df = pd.read_csv(file_name, index_col=0)
            self.create_table_widget(self.eeg_table, df)
            self._chk_data['eeg'] = True

    def _import_mri_data(self):
        file_name = self._import_data_path()
        if file_name:
            df = pd.read_csv(file_name, index_col=0)
            self.create_table_widget(self.mri_table, df)
            self._chk_data['mri'] = True


    def create_table_widget(self, widget, df):
        widget.setRowCount(len(df.index))
        widget.setColumnCount(len(df.columns))
        widget.setHorizontalHeaderLabels(df.columns)
        widget.setVerticalHeaderLabels(df.index.astype('str'))

        for row_index, row in enumerate(df.index):
            for col_index, column in enumerate(df.columns):
                value = df.loc[row][column]
                item = QtWidgets.QTableWidgetItem(str(value))
                widget.setItem(row_index, col_index, item)

    def _show_fig(self,*features,fig):
        ax = fig.add_subplot(111)
        x,y=features
        ax.plot(x,y, label="sin")
        ax.set_xlabel("x")
        ax.set_xlabel("y")

        ax.set_title("my sin graph")
        ax.legend()


    def _show_vr_features(self):
        vr_fig = plt.Figure()
        self.vr_canvas = FigureCanvas(vr_fig)
        self.vr_features_figure.addWidget(self.vr_canvas)

        x = np.arange(0, 100, 1)
        y = np.sin(x)
        self._show_fig(x,y,fig=vr_fig)
        #vr_fig.tight_layout()
        self.vr_canvas.draw()


    def _show_ep_features(self):
        ep_fig = plt.Figure()
        self.ep_canvas = FigureCanvas(ep_fig)
        self.ep_features_figure.addWidget(self.ep_canvas)

        x = np.arange(0, 100, 1)
        y = np.sin(x)
        self._show_fig(x,y,fig=ep_fig)
        self.vr_canvas.draw()

    def _show_eeg_features(self):
        eeg_fig = plt.Figure()
        self.eeg_canvas = FigureCanvas(eeg_fig)
        self.eeg_features_figure.addWidget(self.eeg_canvas)

        x = np.arange(0, 100, 1)
        y = np.sin(x)
        self._show_fig(x,y,fig=eeg_fig)
        self.eeg_canvas.draw()

    def _show_mri_features(self):
        mri_fig = plt.Figure()
        self.mri_canvas = FigureCanvas(mri_fig)
        self.mri_features_figure.addWidget(self.mri_canvas)

        x = np.arange(0, 100, 1)
        y = np.sin(x)
        self._show_fig(x,y,fig=mri_fig)
        self.mri_canvas.draw()

    def _show_vr_result(self):

        vr_result_fig = plt.Figure()
        self.vr_result_canvas = FigureCanvas(vr_result_fig)
        self.vr_result.addWidget(self.vr_result_canvas)

        fig = vr_result_fig

        ax = fig.add_subplot(111)

        group_names = ['HC', 'MCI']
        group_sizes = [95, 32]
        group_colors = ['yellowgreen','lightcoral']
        group_explodes = (0.1, 0)  # explode 1st slice

        ax.pie(
            group_sizes, explode = group_explodes, labels = group_names, colors = group_colors, autopct = '%1.2f%%', shadow=True, startangle=90, textprops={'fontsize': 20})
        ax.set_title('VR', fontsize=40)
        fig.tight_layout()
        self.vr_result_canvas.draw()

    def _show_ep_result(self):

        ep_result_fig = plt.Figure()
        self.ep_result_canvas = FigureCanvas(ep_result_fig)
        self.ep_result.addWidget(self.ep_result_canvas)

        fig = ep_result_fig

        ax = fig.add_subplot(111)

        group_names = ['HC', 'MCI']
        group_sizes = [95, 54]
        group_colors = ['yellowgreen','lightcoral']
        group_explodes = (0.1, 0)  # explode 1st slice

        ax.pie(
            group_sizes, explode = group_explodes, labels = group_names, colors = group_colors, autopct = '%1.2f%%', shadow=True, startangle=90, textprops={'fontsize': 20})
        ax.set_title('EP', fontsize=40)
        fig.tight_layout()
        self.ep_result_canvas.draw()


    def _show_eeg_result(self):

        eeg_result_fig = plt.Figure()
        self.eeg_result_canvas = FigureCanvas(eeg_result_fig)
        self.eeg_result.addWidget(self.eeg_result_canvas)

        fig = eeg_result_fig

        ax = fig.add_subplot(111)

        group_names = ['HC', 'MCI']
        group_sizes = [95, 54]
        group_colors = ['yellowgreen','lightcoral']
        group_explodes = (0.1, 0)  # explode 1st slice

        ax.pie(
            group_sizes, explode = group_explodes, labels = group_names, colors = group_colors, autopct = '%1.2f%%', shadow=True, startangle=90, textprops={'fontsize': 20})
        ax.set_title('EEG', fontsize=40)
        fig.tight_layout()
        self.eeg_result_canvas.draw()

    def _show_mri_result(self):

        mri_result_fig = plt.Figure()
        self.mri_result_canvas = FigureCanvas(mri_result_fig)
        self.mri_result.addWidget(self.mri_result_canvas)

        fig = mri_result_fig

        ax = fig.add_subplot(111)

        group_names = ['HC', 'MCI']
        group_sizes = [180,15]
        group_colors = ['yellowgreen','lightcoral']
        group_explodes = (0.1, 0)  # explode 1st slice

        ax.pie(
            group_sizes, explode = group_explodes, labels = group_names, colors = group_colors, autopct = '%1.2f%%', shadow=True, startangle=90, textprops={'fontsize': 20})
        ax.set_title('MRI', fontsize=40)
        fig.tight_layout()
        self.mri_result_canvas.draw()


    def _show_total_result(self):
        total_fig = plt.Figure()
        self.total_canvas = FigureCanvas(total_fig)
        self.total_figure.addWidget(self.total_canvas)

        fig = total_fig

        ax = fig.add_subplot(111)

        group_names = ['HC', 'MCI']
        group_sizes = [95, 54]

        subgroup_names = ['VR', 'EP', 'EEG','MRI', 'VR', 'EP', 'EEG','MRI']
        subgroup_sizes = [50, 30, 10,5, 30, 10,10, 4]  # colorsa,
        a, b = [plt.cm.Greens,plt.cm.Reds]
        width_num = 0.4

        ax.axis('equal')
        pie_outside, _ = ax.pie(group_sizes, radius=1.3, labels=group_names, labeldistance=0.8,
                                colors=[a(0.6), b(0.6)],textprops={'fontsize': 20})
        plt.setp(pie_outside, width=width_num, edgecolor='white')
        pie_inside, plt_labels, junk = ax.pie(subgroup_sizes, radius=(1.3 - width_num), labels=subgroup_names,
                                              labeldistance=0.75, autopct='%1.1f%%',
                                              colors=[a(0.5), a(0.4), a(0.3),a(0,2), b(0.5), b(0.4), b(0.3),b(0.2)],textprops={'fontsize': 14})
        plt.setp(pie_inside, width=width_num, edgecolor='white')
        ax.set_title('Total', fontsize=40)
        fig.tight_layout()
        self.total_canvas.draw()

class DetailAboutVEEM(QtWidgets.QMainWindow):

    def __init__(self, main_class):
        super().__init__()
        uic.loadUi('./veem_gui/ui/VEEM_popup.ui', self)
        self.main = main_class
        print(self.main.widget.count())
        print(self.main.widget.currentIndex())



    def accept(self) -> None:
        export_path=self.export_path.toPlainText()
        data_save_type=self.comboBox.currentText()
        if any(self.main.data_set_path):
            try:
               self.main.raw_eeg.save(f'{export_path}.fif')
            except (FileNotFoundError, RuntimeWarning):
                self.export_path.setText('해당 주소가 없습니다.')
            else:
                super().accept()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    widget_stack = QtWidgets.QStackedWidget()

    widget_stack.addWidget(Main(widget_stack))
    widget_stack.showNormal()
    app.exec_()








