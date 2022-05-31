import sys
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
import utils
from generator import Generator
import config as cfg
import torch
import cv2
from video_frame_conversion import convert_video_to_frames, convert_frames_to_video
import os

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("video_filter_ui.ui", self)
        self.video_button.clicked.connect(self.browse_videos)
    
    def browse_videos(self):
        self.label_2.setText("")
        fpath = QFileDialog.getOpenFileName(self, "Open File", "","MP4 files (*.mp4)")
        fname = ""
        if(fpath[0] != ""):
            fname = self.file_name_generator(fpath[0])
            output_path = os.getcwd() + "\\" + fname[:len(fname)-4] + "_filtered.mp4"
            self.create_output_video(fpath[0], output_path)
            self.label_2.setText("Output Path: " + output_path)
        
    def create_output_video(self,input_video_path, generated_video_path):
        trained_model = Generator(input_channels=3).to(cfg.device)
        optimizer = torch.optim.Adam(trained_model.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
        utils.load_model_checkpoint("filter_generator_model_100_epoch.pt", trained_model, optimizer, cfg.learning_rate)
        convert_frames_to_video(convert_video_to_frames(cv2.VideoCapture(input_video_path), trained_model), generated_video_path)

    def file_name_generator(self, path):
        i = -1
        file_name = ""
        while(path[i] != "/"):
            file_name = file_name + path[i]
            i = i - 1
        
        return file_name[::-1]

app = QApplication(sys.argv)
mainWindow = MainWindow()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainWindow)
widget.setFixedHeight(320)
widget.setFixedWidth(1200)
widget.show()

try:
    sys.exit(app.exec_())
except: 
    print("Exiting")

        