import os
import random
import numpy as np
import splitfolders     # for slick train/val/test set splitting
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Macros for directories
INPUT_DIR = "Images"
OUTPUT_DIR = "output"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

input_files = sorted(os.listdir(INPUT_DIR))

# Reformat messy file names to "[class_label]_[#].jpg"
def clean_file_names():
    for f in input_files:
        images = sorted(os.listdir(os.path.join(os.getcwd(), INPUT_DIR, f)))
        count = 1
        for img in images:
            clean_name = str(f) + "_" + str(count) + ".jpeg"
            os.rename(os.path.join(os.getcwd(), INPUT_DIR, f, img), os.path.join(os.getcwd(), INPUT_DIR, f, clean_name))
            count += 1

# Split data into training, validation, test sets
def get_data():
    if os.path.isdir(OUTPUT_DIR):
        print("Output path already exists. Skipping data splitting.")
    else:
        splitfolders.ratio(INPUT_DIR, output=OUTPUT_DIR, seed=1234, ratio=(0.7, 0.15, 0.15), group_prefix=None)
    
    train = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
    )

    val = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
    )    

    test = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
        batch_size=1,
    )
    
    # Further split test set into x and y
    X_test, y_test = tuple(zip(*test))
    X_test = np.squeeze(np.array(X_test))
    y_test = np.squeeze(np.array(y_test))

    return train, val, X_test, y_test

# Returns a Tuple of Lists - (list of strings of class labels, list of ints corresponding to first list)
def get_class_labels():
    class_labels = input_files
    class_ints = [i for i in range(len(class_labels))]
    return class_labels, class_ints
