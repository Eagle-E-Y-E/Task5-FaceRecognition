import os
import cv2
import numpy as np

def _to_uint8(img):
    """
    Ensure the image is uint8 for imshow:
      - if it's float in [0,1], scale up,
      - otherwise, cast to uint8.
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def detect_faces(images, extract_all_faces=False):
    """
    Given a list of images, detect faces using OpenCV's Haar cascade.
    If extract_all_faces is False, for each image the largest detected face is used.
    """
    cropped_faces = []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for image in images:
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=5)
        if len(faces) == 0:
            continue

        if not extract_all_faces:
            # pick the largest rectangle
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            crop = image[y:y+h, x:x+w]
            cropped_faces.append(_to_uint8(crop))
        else:
            for (x, y, w, h) in faces:
                crop = image[y:y+h, x:x+w]
                cropped_faces.append(_to_uint8(crop))

    return cropped_faces