import pickle
import numpy as np

import weka.core.jvm as jvm
from weka.core.dataset import Attribute, Instance, Instances
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.serialization import write
from weka.core.converters import Saver


# =========================================================
# CONFIG
# =========================================================

CIFAR_PATH = "cifar-10-batches-py"

MAX_TRAIN_SAMPLES = 10000   # stability first (IMPORTANT)

NUM_TREES = 200
MAX_DEPTH = 25
NUM_FEATURES = 0

CV_FOLDS = 5


# =========================================================
# LOAD CIFAR
# =========================================================

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    return batch[b'data'], batch[b'labels']


def load_label_names():
    with open(f"{CIFAR_PATH}/batches.meta", 'rb') as f:
        meta = pickle.load(f, encoding='bytes')

    return [x.decode("utf-8") for x in meta[b'label_names']]


def load_training_data():
    all_data = []
    all_labels = []

    for i in range(1, 6):
        data, labels = load_batch(f"{CIFAR_PATH}/data_batch_{i}")
        all_data.append(data)
        all_labels.extend(labels)

    X = np.vstack(all_data)
    y = np.array(all_labels)

    return X, y


# =========================================================
# WEKA DATASET CREATION
# =========================================================

def create_weka_dataset(X, y, class_names):

    attributes = []

    # 3072 pixel features
    for i in range(X.shape[1]):
        attributes.append(Attribute.create_numeric(f"pixel_{i}"))

    # class attribute
    class_attr = Attribute.create_nominal("class", class_names)
    attributes.append(class_attr)

    dataset = Instances.create_instances(
        "CIFAR10",
        attributes,
        len(X)
    )

    dataset.class_is_last()

    for features, label in zip(X, y):

        values = list(features)
        values.append(int(label))

        inst = Instance.create_instance(values)
        dataset.add_instance(inst)

    return dataset


# =========================================================
# PREPROCESSING (SAFE)
# =========================================================

def standardize_dataset(dataset):

    from weka.filters import Filter

    standardize = Filter(
        classname="weka.filters.unsupervised.attribute.Standardize"
    )

    standardize.inputformat(dataset)

    return standardize.filter(dataset)


# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(dataset, filename="cifar10_processed.arff"):

    saver = Saver(classname="weka.core.converters.ArffSaver")
    saver.save_file(dataset, filename)

    print(f"Dataset saved to {filename}")


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(dataset):

    classifier = Classifier(
        classname="weka.classifiers.trees.RandomForest",
        options=[
            "-I", str(NUM_TREES),
            "-depth", str(MAX_DEPTH),
            "-K", str(NUM_FEATURES)
        ]
    )

    classifier.build_classifier(dataset)

    return classifier


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(classifier, dataset):

    evaluation = Evaluation(dataset)

    evaluation.crossvalidate_model(
        classifier,
        dataset,
        CV_FOLDS,
        Random(1)
    )

    print("\n=== RESULTS ===")
    print(evaluation.summary())


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(classifier, filename="cifar10.model"):

    write(filename, classifier)
    print(f"Model saved to: {filename}")


# =========================================================
# MAIN
# =========================================================

def main():

    print("Starting JVM...")
    jvm.start(max_heap_size="6g")

    try:

        print("Loading CIFAR...")
        X, y = load_training_data()
        class_names = load_label_names()

        # =====================================================
        # LIMIT DATASET (IMPORTANT FOR STABILITY)
        # =====================================================
        X = X[:MAX_TRAIN_SAMPLES]
        y = y[:MAX_TRAIN_SAMPLES]

        # =====================================================
        # NORMALIZE BEFORE WEKA
        # =====================================================
        X = X / 255.0

        print("Creating WEKA dataset...")
        dataset = create_weka_dataset(X, y, class_names)

        # =====================================================
        # PREPROCESSING
        # =====================================================
        print("Standardizing dataset...")
        dataset = standardize_dataset(dataset)

        dataset.class_is_last()

        # =====================================================
        # SAVE PROCESSED DATASET
        # =====================================================
        save_dataset(dataset)

        # =====================================================
        # TRAIN
        # =====================================================
        print("Training model...")
        classifier = train_model(dataset)

        # =====================================================
        # SAVE MODEL
        # =====================================================
        save_model(classifier)

        # =====================================================
        # EVALUATE
        # =====================================================
        print("Evaluating...")
        evaluate_model(classifier, dataset)

    finally:

        print("Stopping JVM...")
        jvm.stop()


if __name__ == "__main__":
    main()