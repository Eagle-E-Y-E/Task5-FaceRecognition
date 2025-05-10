
from io import StringIO
from scipy import ndimage
from random import randint
import base64
import cv2
import matplotlib
import os
matplotlib.use('Agg')


def detect_faces():
    image=read_rgb(gray=True, normalize=False)
    # Load the pre-trained classifiers for face
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces=face_cascade.detectMultiScale(image, scaleFactor=1.05,minNeighbors=5)

    return faces

def draw_faces():
    # Read the original colored image.
    image = read_rgb(gray=False, normalize=False)

    # Detect faces using a grayscale image internally.
    faces = detect_faces()

    # Create a list to store cropped face images.
    crops = []
    
    # For each detected face, crop the area from the image.
    for (x, y, w, h) in faces: 
        crop = image[y:y+h, x:x+w]
        crops.append(crop)
        
    return crops


def read_rgb(gray=False, normalize=False):
    # Change the path below to your test image's filename.
    img = cv2.imread("Enstien.png")
    if img is None:
        raise FileNotFoundError("The test image could not be found.")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == "__main__":
    face_crops = draw_faces()

    # If there are multiple cropped faces, show them in separate windows.
    for index, crop in enumerate(face_crops):
        cv2.imshow(f"Face {index}", crop)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

