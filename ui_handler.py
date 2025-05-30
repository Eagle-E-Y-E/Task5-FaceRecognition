import sys
from PyQt5 import uic
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
import cv2
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step, show_histogram_on_label, clear_graphics_view
from face_recognition import face_recognition_


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # self.thresholder = Thresholder()
        uic.loadUi('ui.ui', self)
        self.input_image = None

        self.input_img1.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img1)
        # connect buttons
        self.apply_btn.clicked.connect(self.handle_apply)

    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)
        if widget == self.input_img1:
            self.input_image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.input_image = cv2.resize(self.input_image, (64, 64))
            
        if widget == self.input_img_thresholding:
            self.thresholding_image = cv2.imread(self.img_path)


    def handle_apply(self):
        
        # print("Predicted person:", pred_label)
        face_recognition_(query_image_path=self.img_path)
        self.output_img = cv2.imread(self.img_path) 
        ## edit here to draw  rectangle based on faced and show the predicted label on it
        display_image_Graphics_scene(self.output_img1_GV, self.output_img)

                



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
