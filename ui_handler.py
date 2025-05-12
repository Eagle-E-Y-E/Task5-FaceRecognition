import sys
from PyQt5 import uic
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
import cv2
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step, show_histogram_on_label, clear_graphics_view
import joblib
# from face_recognition import recognize_single_face
from FINAL import recognize_faces



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # self.thresholder = Thresholder()
        uic.loadUi('ui.ui', self)
        self.input_image = None
        self.input_image_rec = None

        self.input_img1.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img1)
        self.input_img_Rec.mouseDoubleClickEvent = lambda event: self.doubleClickHandler(
            event, self.input_img_Rec)
        
        # connect buttons
        self.apply_btn.clicked.connect(self.handle_apply)
        self.apply_btn_Rec.clicked.connect(self.handle_recognize)

        # sliders
        self.num_pca_slider.valueChanged.connect(
            lambda : self.num_pca_label.setText(f"{self.num_pca_slider.value()}"))
        self.threshold_slider.valueChanged.connect(
            lambda : self.threshold_label.setText(f"{self.threshold_slider.value()}"))

    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)
        if widget == self.input_img1:
            self.input_image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.input_image = cv2.resize(self.input_image, (64, 64))
        if widget == self.input_img_Rec:
            self.input_image_rec = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.input_image_rec = cv2.resize(self.input_image_rec, (64, 64))
            
            


    def handle_apply(self):
        print("Apply button clicked")
        
        # print("Predicted person:", pred_label)
        # face_recognition_(query_image_path=self.img_path)
        # self.output_img = cv2.imread(self.img_path) 
        # ## edit here to draw  rectangle based on faced and show the predicted label on it
        # display_image_Graphics_scene(self.output_img1_GV, self.output_img)

    def handle_recognize(self):
        pred,dist = recognize_faces(single_img_path=self.img_path,
                        num_components=self.num_pca_slider.value(),
                        threshold=self.threshold_slider.value())
        self.reult_label.setText(pred)
        self.distance_label.setText(f"distance: {dist:.2f}")
        

                



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
