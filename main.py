import pickle
import numpy as np

import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

CIFAR_PATH = "cifar-10-batches-py"

# =========================================================
# LOAD CIFAR FILE
# =========================================================

def load_batch(file_path):

    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    data = batch[b'data']
    labels = batch[b'labels']

    return data, labels


# =========================================================
# LOAD META LABELS
# =========================================================

def load_label_names():

    with open(f"{CIFAR_PATH}/batches.meta", 'rb') as f:
        meta = pickle.load(f, encoding='bytes')

    labels = [x.decode("utf-8") for x in meta[b'label_names']]

    return labels


# =========================================================
# LOAD ALL TRAINING DATA
# =========================================================

def load_training_data():

    all_data = []
    all_labels = []

    for i in range(1, 6):

        data, labels = load_batch(
            f"{CIFAR_PATH}/data_batch_{i}"
        )

        all_data.append(data)
        all_labels.extend(labels)

    X = np.vstack(all_data)
    y = np.array(all_labels)

    return X, y


# =========================================================
# CREATE WEKA DATASET
# =========================================================

def create_weka_dataset(X, y, class_names):

    attributes = []

    # 3072 pixel features
    for i in range(3072):
        attributes.append(
            Attribute.create_numeric(f"pixel_{i}")
        )

    # Class labels
    class_attr = Attribute.create_nominal(
        "class",
        class_names
    )

    attributes.append(class_attr)

    dataset = Instances.create_instances(
        "CIFAR10",
        attributes,
        len(X)
    )

    dataset.class_is_last()

    # Add data rows
    for features, label in zip(X, y):

        # Normalize pixels
        features = features / 255.0

        values = list(features)

        # Append class index
        values.append(int(label))

        inst = Instance.create_instance(values)

        dataset.add_instance(inst)

    return dataset


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(dataset):

    classifier = Classifier(
        classname="weka.classifiers.trees.RandomForest"
    )

    classifier.build_classifier(dataset)

    return classifier


# =========================================================
# EVALUATE
# =========================================================

def evaluate_model(classifier, dataset):

    evaluation = Evaluation(dataset)

    evaluation.crossvalidate_model(
        classifier,
        dataset,
        5,
        Random(1)
    )

    print(evaluation.summary())


# =========================================================
# MAIN
# =========================================================

def main():

    print("Starting JVM...")
    jvm.start()

    try:

        print("Loading labels...")
        class_names = load_label_names()

        print("Classes:", class_names)

        print("Loading CIFAR data...")
        X, y = load_training_data()

        print("Dataset shape:", X.shape)

        print("Creating WEKA dataset...")
        dataset = create_weka_dataset(
            X,
            y,
            class_names
        )

        print("Training model...")
        classifier = train_model(dataset)

        print("Evaluating...")
        evaluate_model(classifier, dataset)

    finally:

        print("Stopping JVM...")
        jvm.stop()


if __name__ == "__main__":
    main()
