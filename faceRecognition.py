import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pca import PCA_

def recognize_faces(
    train_dir="orl_faces_train",
    test_dir="orl_faces_test",
    single_img_path=None,
    image_size=(64, 64),
    num_components=100,
    threshold=4000
):
    def load_images_labels(folder, image_size=(64, 64)):
        X, y, paths = [], [], []
        for label in sorted(os.listdir(folder)):
            person_dir = os.path.join(folder, label)
            if not os.path.isdir(person_dir):
                continue
            for fname in os.listdir(person_dir):
                fpath = os.path.join(person_dir, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, image_size)
                X.append(img_resized.flatten())
                y.append(label)
                paths.append(fpath)
        return np.array(X), np.array(y), paths


    def predict_single_image(img_path, pca_, X_train_pca_, y_train_enc, le, train_paths, image_size, threshold=4000):
        # Load and preprocess the query image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)
        img_flat = img.flatten().reshape(1, -1)
        img_pca = pca_.transform(img_flat)
        
        # Compute distances in the reduced PCA space
        dists = np.sqrt(np.sum((X_train_pca_ - img_pca) ** 2, axis=1))
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist < threshold:
            pred_label_enc = y_train_enc[min_idx]
            pred_label = le.inverse_transform([pred_label_enc])[0]
            # Load the similar image from disk using the stored path, then resize appropriately
            similar_img = cv2.imread(train_paths[min_idx], cv2.IMREAD_GRAYSCALE)
            similar_img = cv2.resize(similar_img, image_size)
        else:
            pred_label = "unknown"
            similar_img = cv2.imread("images/unknown.jpg")
        
        return pred_label, min_dist, similar_img


    # Load the training and test data along with file paths for training images
    X_train, y_train, train_paths = load_images_labels(train_dir, image_size)
    X_test, y_test, _ = load_images_labels(test_dir, image_size)

    # Encode labels for training/testing
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # PCA transform on training data
    pca_ = PCA_(num_components=num_components)
    pca_.fit(X_train)
    X_train_pca_ = pca_.transform(X_train)
    X_test_pca_ = pca_.transform(X_test)

    # Test set evaluation (Euclidean NN using threshold)
    y_pred_euclid_thresh = []
    for test_vec in X_test_pca_:
        dists = np.sqrt(np.sum((X_train_pca_ - test_vec) ** 2, axis=1))
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        if min_dist < threshold:
            y_pred_euclid_thresh.append(y_train_enc[min_idx])
        else:
            y_pred_euclid_thresh.append(-1)
    y_pred_euclid_thresh = np.array(y_pred_euclid_thresh)
    mask_known = y_pred_euclid_thresh != -1
    if np.any(mask_known):
        acc_known = accuracy_score(y_test_enc[mask_known], y_pred_euclid_thresh[mask_known])
        print(f"Test accuracy (PCA_ + Euclidean NN, threshold={threshold}): {acc_known:.2%}")
    else:
        print("No known predictions (all classified as unknown).")
    print(f"Unknown predictions: {(~mask_known).sum()} out of {len(y_pred_euclid_thresh)}")

    # Predict a single query image and return the similar training image if provided
    if single_img_path:
        pred, dist, similar_img = predict_single_image(
            single_img_path, pca_, X_train_pca_, y_train_enc, le, train_paths, image_size, threshold
        )
        print(f"Predicted label: {pred}, Distance: {dist}")
        return pred, dist, similar_img

    return