import os
import random
import numpy as np
import splitfolders     # for slick train/val/test set splitting
import tensorflow as tf
from tensorflow import keras

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
        images = os.listdir(os.path.join(os.getcwd(), INPUT_DIR, f))
        count = 1
        for img in images:
            clean_name = str(f) + "_" + str(count) + ".jpg"
            os.rename(os.path.join(os.getcwd(), INPUT_DIR, f, img), os.path.join(os.getcwd(), INPUT_DIR, f, clean_name))
            count += 1

# Split data into training, validation, test sets
def get_data():
    if os.path.isdir(OUTPUT_DIR):
        print("Output path already exists. Skipping data splitting.")
    else:
        splitfolders.ratio(INPUT_DIR, output=OUTPUT_DIR, seed=1234, ratio=(0.8, 0.1, 0.1), group_prefix=None)
    
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
    )

    train_data = list(train)[0]
    val_data = list(val)[0]
    test_data = list(test)[0]
    
    # Further split test set into x and y
    X_test = test_data[0]
    y_test = test_data[1]

    return train_data, val_data, X_test, y_test