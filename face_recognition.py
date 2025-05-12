import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from faceDetection import detect_faces
# ----------------------------------------------------

def read_images(dataset_path, image_size=(100, 100), detect=True):
    """
    Read images from dataset_path where each subfolder corresponds to a subject.
    If detect==True, each image is processed through a face detector.
    Returns images as flattened arrays, labels, and a mapping from label id to subject name.
    """    
    images = []
    labels = []
    label_names = {}
    
    # Start with label_id 0 for known subjects.
    # Impostor (unknown) faces will be given a label of -1.
    for subject in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path, subject)
        if os.path.isdir(subject_path):
            # Check if the folder name equals "-1". If so, assign label -1.
            if subject == "-1":
                subject_label = -1
            else:
                # For other subjects, assign a sequential label.
                subject_label = len(label_names)
                label_names[subject_label] = subject
                
            for filename in os.listdir(subject_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.pgm')):
                    continue
                img_path = os.path.join(subject_path, filename)
                # Read the image in grayscale
                raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if raw_img is None:
                    continue

                # If detection is enabled, crop the face before resizing.
                if detect:
                    faces = detect_faces([raw_img], extract_all_faces=False)
                    if len(faces) == 0:
                        continue  # no face found; skip image
                    face = faces[0]
                else:
                    face = raw_img
                
                # Resize the face to the desired size and flatten it.
                face_resized = cv2.resize(face, image_size)
                images.append(face_resized.flatten())
                labels.append(subject_label)
    print(f"Loaded a total of {len(images)} images from {dataset_path}")
    return np.array(images, dtype="float32"), np.array(labels), label_names


# ----------------------------------------------------
# PCA, projection, and recognition functions
# (same as before)
# ----------------------------------------------------
def compute_pca(X, num_components):
    """
    Compute PCA without using built-in PCA functions.
    X: matrix where each row is a flattened image.
    Returns the mean face, selected eigenvalues, and eigenvectors (eigenfaces).
    """
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    num_samples = X_centered.shape[0]
    # Compute a small covariance matrix (M x M, where M is num_samples)
    L = np.dot(X_centered, X_centered.T) / num_samples

    eigenvalues, eigenvectors_small = np.linalg.eig(L)
    eigenvectors = np.dot(X_centered.T, eigenvectors_small)
    
    # Normalize eigenvectors (each column is an eigenface)
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = eigenvalues[:num_components]
    eigenvectors = eigenvectors[:, :num_components]
    
    return mean_face, eigenvalues, eigenvectors

def project_face(face, mean_face, eigenvectors):
    """Project a face onto the PCA subspace."""
    face_centered = face - mean_face
    projection = np.dot(face_centered, eigenvectors)
    return projection

def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

def face_recognition(train_images, train_labels, test_images, test_labels, mean_face, eigenvectors, threshold):
    """
    Recognize faces by projecting them onto the PCA subspace and comparing.
    A test image is assigned the label of the closest training image, provided
    that distance is below a threshold; otherwise, it is marked unknown (-1).
    """
    correct = 0
    distances = []
    labels_pred = []
    
    # Precompute projections for training images.
    train_projections = [project_face(face, mean_face, eigenvectors) for face in train_images]
    
    for i, test_face in enumerate(test_images):
        proj_test = project_face(test_face, mean_face, eigenvectors)
        dists = [euclidean_distance(proj_test, train_proj) for train_proj in train_projections]
        min_index = np.argmin(dists)
        min_dist = dists[min_index]
        distances.append(min_dist)
        
        if min_dist < threshold:
            pred = train_labels[min_index]
        else:
            pred = -1
        labels_pred.append(pred)
        
        if pred == test_labels[i]:
            correct += 1
            
    accuracy = correct / len(test_images)
    return accuracy, distances, labels_pred

def compute_roc(distances, true_labels, thresholds):
    """
    Compute ROC by iterating over thresholds.
    For each test sample, if the distance is below threshold it is accepted.
    We compute true positive rate (TPR) and false positive rate (FPR) accordingly.
    """
    tpr_list = []
    fpr_list = []
    for thr in thresholds:
        TP = FP = TN = FN = 0
        for i, dist in enumerate(distances):
            if dist < thr:  # accepted as known face
                if true_labels[i] != -1:
                    TP += 1
                else:
                    FP += 1
            else:  # rejected
                if true_labels[i] != -1:
                    FN += 1
                else:
                    TN += 1
        tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

def plot_roc_curve(fpr_list, tpr_list):
    """Plot ROC curve using OpenCV drawing functions."""
    width = 500
    height = 500
    roc_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw axes (leaving a margin of 50 pixels)
    cv2.line(roc_img, (50, height - 50), (width - 50, height - 50), (0, 0, 0), 2)  # X-axis
    cv2.line(roc_img, (50, height - 50), (50, 50), (0, 0, 0), 2)  # Y-axis
    
    pts = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        x = int(50 + fpr * (width - 100))
        y = int(height - 50 - tpr * (height - 100))
        pts.append((x, y))
    
    for i in range(len(pts) - 1):
        cv2.line(roc_img, pts[i], pts[i+1], (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
    cv2.line(roc_img, (50, height - 50), (width - 50, 50), (0, 0, 255), 1)

    cv2.imshow("ROC Curve", roc_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# (Include your helper functions: _to_uint8, detect_faces, read_images, compute_pca,
# project_face, euclidean_distance, and face_recognition from your code above.)

# ------------------------------------------------------------------------------
# New function: recognize_query_image(...)
# ------------------------------------------------------------------------------
def recognize_query_image(query_img_path, train_images, train_labels, mean_face, eigenvectors, threshold, label_names, image_size=(100, 100)):
    """
    Reads a query image, detects (and crops) its face, projects it into the PCA
    subspace, and then finds the closest matching training image.
    It then overlays the predicted label on the query image and, if a match is found,
    displays both the query and match side-by-side.
    """
    # Read the query image in grayscale
    raw_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print("Could not read query image.")
        return

    # Detect face(s) in the query image
    query_faces = detect_faces([raw_img], extract_all_faces=False)
    if len(query_faces) == 0:
        print("No face detected in the query image.")
        return

    # Use the largest/first detected face and resize it.
    query_face = query_faces[0]
    query_face_resized = cv2.resize(query_face, image_size)

    # Flatten the query face for processing.
    query_flat = query_face_resized.flatten().astype("float32")

    # Project the query face onto the PCA subspace.
    query_projection = project_face(query_flat, mean_face, eigenvectors)

    # Precompute training projections if not already done.
    train_projections = [project_face(t_face, mean_face, eigenvectors) for t_face in train_images]

    # Compute distances between query projection and all training projections.
    dists = [euclidean_distance(query_projection, proj) for proj in train_projections]
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]

    # Decide whether we recognize the face based on the threshold.
    if min_dist < threshold:
        predicted_label = train_labels[min_idx]
        predicted_name = label_names[predicted_label]
        # Get the matching training image for display (reshape as needed).
        match_flat = train_images[min_idx]
        match_img = match_flat.reshape(image_size)
    else:
        predicted_label = -1
        predicted_name = "Unknown"
        match_img = None

    # Prepare the query image for visualization (convert gray to BGR for colored text)
    query_disp = cv2.cvtColor(query_face_resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(query_disp, predicted_name, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if match_img is not None:
        match_disp = cv2.cvtColor(match_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(match_disp, predicted_name, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Combine query and match images side-by-side
        combined = np.hstack((query_disp, match_disp))
        cv2.imshow("Query (left) and Best Match (right)", combined)
        cv2.imwrite("query_and_match.png", combined)  # Optionally save to file.
    else:
        cv2.imshow("Query Face", query_disp)
        cv2.imwrite("query_unknown.png", query_disp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_recognition_(query_image_path="Enstien.png", train_path = "orl_faces_train", test_path = "orl_faces_test",  num_components = 100):
    # Load training data from the training folder (with face detection enabled)
    train_images, train_labels, train_label_names = read_images(train_path, image_size=(100, 100), detect=False)
    # Load testing data from the test folder (with face detection enabled)
    test_images, test_labels, test_label_names = read_images(test_path, image_size=(100, 100), detect=False)

    # Compute PCA on the training images
    mean_face, eigenvalues, eigenvectors = compute_pca(train_images, num_components)

    # Set a recognition threshold (tuning may be required based on your data)
    threshold = 4000

    accuracy, distances, predicted_labels = face_recognition(
        train_images, train_labels, test_images, test_labels, mean_face, eigenvectors, threshold
    )
    print("Recognition Accuracy =", accuracy)

    # Call the function to recognize the query image.
    recognize_query_image(query_image_path, train_images, train_labels, mean_face, eigenvectors, threshold, train_label_names, image_size=(100, 100))

    # Compute and plot the ROC curve
    min_thr = np.min(distances)
    max_thr = np.max(distances)
    thresholds = np.linspace(min_thr, max_thr, num=100)
    tpr_list, fpr_list = compute_roc(distances, test_labels, thresholds)
    plot_roc_curve(fpr_list, tpr_list)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, marker='o', color='blue', linewidth=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--', label='Chance')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.show()