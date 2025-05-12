from io import StringIO
from random import randint
import base64
import cv2
import matplotlib
import os
matplotlib.use('Agg')


def detect_faces():
    # Use a grayscale image for detection.
    image = read_rgb(gray=True, normalize=False)
    # Load the pre-trained classifier for faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5)
    return faces


def draw_faces():
    # Read the original colored image.
    image = read_rgb(gray=False, normalize=False)
    # Detect faces using the grayscale version internally.
    faces = detect_faces()

    # For each detected face, draw a rectangle on the image.
    for (x, y, w, h) in faces:
        # Draw a green rectangle with thickness 2.
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


def read_rgb(gray=False, normalize=False):
    # Change the path to your test image.
    img = cv2.imread("pepsi_can.png")
    if img is None:
        raise FileNotFoundError("The test image could not be found.")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == "__main__":
    # Get the image with drawn squares around detected faces.
    annotated_image = draw_faces()

    cv2.imshow("Faces with Borders", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
