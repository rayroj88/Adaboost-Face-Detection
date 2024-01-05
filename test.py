import cv2
import os
from joblib import load
from config import training_results_directory, data_directory
from train import StrongClassifier, build_cascade, detect_faces, get_trained_cascade
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

def test_face_detection(image, cascade):
    if image is None:
        print("Error: Image not found.")
        return

    # Assuming the image is already in grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the window size and step size
    window_size = 24  # Example value
    step_size = 4     # Example value

    # Detect faces
    detected_faces = detect_faces(grayscale_image, cascade, window_size, step_size)

    # Draw rectangles around detected faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_faces_from_folder(subfolder):
    folder_path = os.path.join(data_directory, subfolder)
    images = []
    for filename in os.listdir(folder_path):
        try:
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                # Apply any necessary preprocessing as in train.py
                preprocessed_img = preprocess_image(img)  # Implement preprocess_image as needed
                images.append(preprocessed_img)
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
                # Apply any necessary preprocessing as in train.py
                preprocessed_img = preprocess_image(img)  # Implement preprocess_image as needed
                images.append(preprocessed_img)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images

def test_folder_of_images(folder_path, cascade):
    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        test_face_detection(image_path, cascade)

def load_trained_cascade(model_filename):
    model_path = os.path.join(training_results_directory, model_filename)
    try:
        return load(model_path)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

def preprocess_image(image):
    # Apply preprocessing steps here
    # For example, resize the image to a specific size
    resized_image = cv2.resize(image, (100, 100))
    return resized_image

if __name__ == "__main__":
    model_filename = 'training_results.joblib'
    cascade = load_trained_cascade(model_filename)

    if cascade is not None:
        # Load test images
        test_faces = load_faces_from_folder('test_face_photos')
        test_non_faces = load_nonfaces_from_folder('test_nonfaces')

        # Run face detection on test faces
        print("Detecting faces in test face images...")
        for test_image_path in test_faces:
            test_face_detection('test_face_photos', cascade)

        # Run face detection on test non-faces
        print("Detecting faces in test non-face images...")
        for test_image_path in test_non_faces:
            test_face_detection('test_nonfaces', cascade)