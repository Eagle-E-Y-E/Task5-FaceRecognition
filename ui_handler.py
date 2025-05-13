import sys
from PyQt5 import uic
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
import cv2
from utils import load_pixmap_to_label, display_image_Graphics_scene, enforce_slider_step, show_histogram_on_label, clear_graphics_view
from faceRecognition import recognize_faces
from faceDetection import draw_faces



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
        self.apply_btn.clicked.connect(self.handle_detect)
        self.apply_btn_Rec.clicked.connect(self.handle_recognize)

        # sliders
        self.num_pca_slider.valueChanged.connect(
            lambda : self.num_pca_label.setText(f"{self.num_pca_slider.value()}"))
        self.threshold_slider.valueChanged.connect(
            lambda : self.threshold_label.setText(f"{self.threshold_slider.value()}"))
        
        self.min_neighbors_slider.valueChanged.connect(
            lambda : self.min_neighbors_label.setText(f"{self.min_neighbors_slider.value()}"))
        self.scale_factor_slider.valueChanged.connect(
            lambda : self.scale_factor_label.setText(f"{self.scale_factor_slider.value()/100}"))
        
    

    def doubleClickHandler(self, event, widget):
        self.img_path = load_pixmap_to_label(widget)
        if widget == self.input_img1:
            self.input_image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.input_image = cv2.resize(self.input_image, (64, 64))
        if widget == self.input_img_Rec:
            self.input_image_rec = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.input_image_rec = cv2.resize(self.input_image_rec, (64, 64))
            
            


    def handle_detect(self):
        print("Apply button clicked")
        print("scale_factor:", self.scale_factor_slider.value()/100)
        print("min_neighbors:", self.min_neighbors_slider.value())
        annotated_img = draw_faces(self.img_path,  self.min_neighbors_slider.value(),self.scale_factor_slider.value()/100)
        
        display_image_Graphics_scene(self.output_img1_GV, annotated_img)

    def handle_recognize(self):
        # Get prediction, distance, and the similar training image
        pred, dist, sim_img = recognize_faces(
            single_img_path=self.img_path,
            num_components=self.num_pca_slider.value(),
            threshold=self.threshold_slider.value()
        )
        self.reult_label.setText(pred)
        self.distance_label.setText(f"Distance: {dist:.2f}")
        
        # Display the similar image if available.
        # Here we assume you have a function to display images in your GUI.
        if sim_img is not None:
            display_image_Graphics_scene(self.output_img_Rec, sim_img)

        

                



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
