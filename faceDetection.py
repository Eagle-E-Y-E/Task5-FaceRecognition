
from io import StringIO
from scipy import ndimage
from random import randint
import base64
import cv2
import matplotlib
import numpy as np
matplotlib.use('Agg')

def read_rgb(image_path, gray=False, normalize=False):
    # Change the path to your test image.
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("The test image could not be found.")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def detect_faces(image_input):
    """
    Detect faces in an image from either a file path or an image array.
    """
    # If the input is a string, assume it's a file path.
    if isinstance(image_input, str):
        gray_image = read_rgb(image_input, gray=True, normalize=False)
    # If the input is a numpy array, assume it's already loaded.
    elif isinstance(image_input, np.ndarray):
        # If the image is colored, convert it to grayscale.
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            gray_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            # It is presumably already grayscale.
            gray_image = image_input
    else:
        raise ValueError("Input must be a file path (str) or an image array (numpy.ndarray).")
    
    
    # Load the pre-trained classifier for faces.
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for image in images:
        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.05, minNeighbors=5
        )

        if len(faces) == 0:
            continue

        if not extract_all_faces:
            # pick the largest rectangle
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            crop = image[y:y+h, x:x+w]
            cropped_faces.append(_to_uint8(crop))
        else:
            for (x, y, w, h) in faces:
                crop = image[y:y+h, x:x+w]
                cropped_faces.append(_to_uint8(crop))

    return cropped_faces


def _to_uint8(img):
    """
    Ensure the image is uint8 for imshow:
      - if it's float in [0,1], scale up
      - if it's any other type, just cast
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def read_rgb(folder_path, gray=False, normalize=False):
    image_list = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.pgm')):
            continue

        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read image {file_path}")
            continue

        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if normalize:
            img = img / 255.0

        image_list.append(img)

    return image_list



if __name__ == "__main__":
    images = read_rgb("test", gray=True, normalize=False)
    face_crops = detect_faces(images)

    # If there are multiple cropped faces, show them in separate windows.
    for index, crop in enumerate(face_crops):
        cv2.imshow(f"Face {index}", crop)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

