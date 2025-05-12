import cv2
import matplotlib
matplotlib.use('Agg')

def read_rgb(image_path, gray=False, normalize=False):
    # Change the path to your test image.
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("The test image could not be found.")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def detect_faces(image_path):
    # Use a grayscale image for detection.
    gray_image = read_rgb(image_path, gray=True, normalize=False)
    # Load the pre-trained classifier for faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5)
    return faces

def draw_faces(image_path):
    # Read the original colored image.
    colored_image = read_rgb(image_path, gray=False, normalize=False)
    # Detect faces using the grayscale version.
    faces = detect_faces(image_path)

    # For each detected face, draw a rectangle on the image.
    for (x, y, w, h) in faces:
        cv2.rectangle(colored_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return colored_image

if __name__ == "__main__":
    image_path = "Enstien.png"  # Change this to your test image path
    annotated_image = draw_faces(image_path=image_path)

    cv2.imshow("Faces with Borders", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
