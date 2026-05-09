import os
import cv2
import numpy as np

import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "dataset"
IMAGE_SIZE = (64, 64)

# =========================================================
# IMAGE FEATURE EXTRACTION
# =========================================================

def extract_features(image_path):
    """
    Extract features from an image.

    Current method:
    - Resize image
    - Flatten RGB values

    Replace with:
    - HOG
    - SIFT
    - CNN embeddings
    - Color histograms
    """
    
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, IMAGE_SIZE)

    # Normalize pixel values
    image = image / 255.0

    # Flatten into 1D feature vector
    features = image.flatten()

    return features


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(dataset_path):
    """
    Loads images and labels from folder structure.
    """

    X = []
    y = []

    class_names = sorted(os.listdir(dataset_path))

    for class_name in class_names:

        class_folder = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_folder):
            continue

        for filename in os.listdir(class_folder):

            image_path = os.path.join(class_folder, filename)

            features = extract_features(image_path)

            if features is not None:
                X.append(features)
                y.append(class_name)

    return np.array(X), np.array(y), class_names


# =========================================================
# CONVERT TO WEKA DATASET
# =========================================================

def create_weka_dataset(X, y, class_names):
    """
    Convert NumPy arrays into WEKA Instances.
    """

    attributes = []

    # Numeric feature attributes
    for i in range(X.shape[1]):
        attributes.append(Attribute.create_numeric(f"pixel_{i}"))

    # Class attribute
    class_attr = Attribute.create_nominal("class", class_names)
    attributes.append(class_attr)

    dataset = Instances.create_instances(
        "ImageClassificationDataset",
        attributes,
        len(X)
    )

    dataset.class_is_last()

    # Add instances
    for features, label in zip(X, y):

        values = list(features)

        # Append class index
        values.append(class_names.index(label))

        inst = Instance.create_instance(values)
        dataset.add_instance(inst)

    return dataset


# =========================================================
# TRAIN MODEL
# =========================================================

def train_classifier(dataset):
    """
    Train WEKA classifier.
    """

    classifier = Classifier(
        classname="weka.classifiers.trees.RandomForest"
    )

    classifier.build_classifier(dataset)

    return classifier


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(classifier, dataset):
    """
    Evaluate classifier using cross-validation.
    """

    evaluation = Evaluation(dataset)

    evaluation.crossvalidate_model(
        classifier,
        dataset,
        10,
        Random(1)
    )

    print("\n=== Evaluation Results ===")
    print(evaluation.summary())
    print(evaluation.class_details())


# =========================================================
# PREDICT SINGLE IMAGE
# =========================================================

def predict_image(classifier, image_path, dataset, class_names):

    features = extract_features(image_path)

    if features is None:
        print("Failed to load image.")
        return

    values = list(features)

    # Placeholder class value
    values.append(0)

    inst = Instance.create_instance(values)
    inst.dataset = dataset

    prediction_index = int(classifier.classify_instance(inst))

    predicted_class = class_names[prediction_index]

    print(f"Prediction: {predicted_class}")


# =========================================================
# MAIN
# =========================================================

def main():

    print("Starting JVM...")
    jvm.start()

    try:

        print("Loading dataset...")
        X, y, class_names = load_dataset(DATASET_PATH)

        print(f"Loaded {len(X)} images")
        print(f"Classes: {class_names}")

        print("Creating WEKA dataset...")
        dataset = create_weka_dataset(X, y, class_names)

        print("Training classifier...")
        classifier = train_classifier(dataset)

        print("Evaluating model...")
        evaluate_model(classifier, dataset)

        # Example prediction
        test_image = "test.jpg"

        if os.path.exists(test_image):
            predict_image(
                classifier,
                test_image,
                dataset,
                class_names
            )

    finally:
        print("Stopping JVM...")
        jvm.stop()


if __name__ == "__main__":
    main()