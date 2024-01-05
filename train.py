import cv2
import os
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from joblib import dump
from config import training_results_directory, data_directory
from boosting import rectangle_filter1
from boosting import rectangle_filter2
from boosting import rectangle_filter3
from boosting import rectangle_filter4
from boosting import rectangle_filter5
from boosting import integral_image
from boosting import generate_classifier1
from boosting import generate_classifier
from boosting import eval_weak_classifier
from boosting import weighted_error
from boosting import find_best_classifier
from boosting import adaboost
from boosting import boosted_predict

class StrongClassifier:
    def __init__(self):
        self.weak_classifiers = []
        self.alphas = []
        self.threshold = 0

    def classify(self, integral_image):
        total = sum(alpha * eval_weak_classifier(wc, integral_image) for alpha, wc in zip(self.alphas, self.weak_classifiers))
        return 1 if total >= self.threshold else -1

def calculate_threshold(weak_classifiers, alphas, integral_images, labels):
    threshold = 0
    max_accuracy = 0

    # Test different threshold values
    for potential_threshold in np.linspace(-1, 1, 20):  # Adjust the range as needed
        correct_predictions = 0
        total_predictions = len(integral_images)

        for image, label in zip(integral_images, labels):
            total = sum(alpha * eval_weak_classifier(wc, image) for alpha, wc in zip(alphas, weak_classifiers))
            prediction = 1 if total >= potential_threshold else -1
            if prediction == label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            threshold = potential_threshold

    return threshold
   
# Function to build the cascade
def build_cascade(integral_images, labels, num_stages, num_weak_classifiers):
    cascade = []
    negatives = integral_images[labels == -1]

    for i in range(num_stages):
        print(f"Training stage {i+1}")

        # Generate weak classifiers for this stage
        weak_classifiers = [generate_classifier(face_vertical, face_horizontal) for _ in range(num_weak_classifiers)]

        # Initialize and populate the responses matrix
        responses = np.zeros((len(integral_images), num_weak_classifiers))
        for j, classifier in enumerate(weak_classifiers):
            for k, image in enumerate(integral_images):
                responses[k, j] = eval_weak_classifier(classifier, image)

        # Define the number of boosting rounds (adjust based on your requirements)
        rounds = 10

        # Train AdaBoost with the populated responses matrix
        ada_results = adaboost(responses, labels, rounds)

        # Create a StrongClassifier instance and populate it with AdaBoost results
        stage_classifier = StrongClassifier()
        stage_classifier.weak_classifiers = [weak_classifiers[idx] for idx, _, _ in ada_results]
        stage_classifier.alphas = [alpha for _, alpha, _ in ada_results]
        stage_classifier.threshold = calculate_threshold(stage_classifier.weak_classifiers, stage_classifier.alphas, integral_images, labels)
        cascade.append(stage_classifier)

        # Update negatives based on the current stage of the cascade
        negatives = [img for img in negatives if cascade_classify(img, cascade) == -1]

    return cascade

# Function to classify using the cascade
def cascade_classify(image, cascade):
    integral = integral_image(image)
    for stage in cascade:
        if stage.classify(integral) == -1:
            return -1  # Not a face
    return 1  # Face

def load_faces_from_folder(subfolder):
    folder_path = os.path.join(data_directory, subfolder)
    images = []
    for filename in os.listdir(folder_path):
        try:
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                skin_img = detect_skin(img)
                resized_img = cv2.resize(skin_img, (100, 100), interpolation=cv2.INTER_AREA)
                images.append(resized_img)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images

def load_nonfaces_from_folder(subfolder):
    folder_path = os.path.join(data_directory, subfolder)
    images = []
    for filename in os.listdir(folder_path):
        try:
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                skin_img = detect_skin(img)
                resized_img = cv2.resize(skin_img, (100, 100), interpolation=cv2.INTER_AREA)
                images.append(resized_img)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images

def detect_skin(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a skin mask
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Bitwise-AND mask and original image
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    return cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

def detect_faces(image, cascade, window_size, step_size):
    detected_faces = []
    img_height, img_width = image.shape

    # Sliding window:
    for y in range(0, img_height - window_size + 1, step_size):
        for x in range(0, img_width - window_size + 1, step_size):
            # Extract the window from the image
            window = image[y:y + window_size, x:x + window_size]

            # Resize window to match the input size expected by the cascade
            resized_window = cv2.resize(window, (face_horizontal, face_vertical))

            # Apply the cascade classifier to the window
            if cascade_classify(resized_window, cascade) == 1:
                detected_faces.append((x, y, window_size, window_size))

    return detected_faces

# Save cascade as a global variable
def get_trained_cascade():
    return cascade

# Path to the datasets
faces_path = os.path.join(data_directory, 'training_faces')
non_faces_path = os.path.join(data_directory, 'training_nonfaces')

faces = []
non_faces = []

# Load face and non-face images
faces = load_faces_from_folder('training_faces')  # 'training_faces' is a subfolder name
non_faces = load_nonfaces_from_folder('training_nonfaces')  # 'training_nonfaces' is a subfolder name

face_vertical = 31
face_horizontal = 25

# Generating weak classifiers
num_weak_classifiers = 1000
weak_classifiers = [generate_classifier(face_vertical, face_horizontal) for _ in range(num_weak_classifiers)]

 # Prepare integral images and labels
integral_faces = [integral_image(face) for face in faces]
integral_non_faces = [integral_image(non_face) for non_face in non_faces]
integral_images = np.concatenate((integral_faces, integral_non_faces), axis=0)
labels = np.array([1] * len(faces) + [-1] * len(non_faces))

if np.all(labels == labels[0]):
    raise ValueError("Error: All labels are the same. Check the dataset.")

# Build the cascade
cascade = build_cascade(integral_images, labels, num_stages=5, num_weak_classifiers=num_weak_classifiers)

# Create the directory for saving the model if it does not exist
if not os.path.exists(training_results_directory):
    os.makedirs(training_results_directory)

# Define the full path for saving the model
model_save_path = os.path.join(training_results_directory, 'training_results.joblib')

# Save the model
dump(cascade, model_save_path)
print(f"Model successfully saved to {model_save_path}")
