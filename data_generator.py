import numpy as np
import os
import pathlib
import pickle
from PIL import Image
import random
import shutil
import splitfolders     # for slick train/val/test set splitting
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Macros for directories
INPUT_DIR = "Images"
OUTPUT_DIR = "output"
AUGMENT_DIR = os.path.join(OUTPUT_DIR, "augmented")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
SEED = 1234

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


# If it doesn't exist yet, split the data into train, validation, and test images
def make_output_dir():
    if os.path.isdir(OUTPUT_DIR):
        print("Output path already exists. Skipping data splitting.")
    else:
        splitfolders.ratio(INPUT_DIR, output=OUTPUT_DIR, seed=SEED, ratio=(0.7, 0.15, 0.15), group_prefix=None)


# Create training, validation, test sets WITHOUT augmentation
def get_data():
    make_output_dir()

    train = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
        seed=SEED,
        batch_size=16,
    )

    val = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
        seed=SEED,
        batch_size=16,
    )    

    test = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
        batch_size=1,
        seed=SEED,
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train = train.cache().prefetch(buffer_size=AUTOTUNE)
    val = val.cache().prefetch(buffer_size=AUTOTUNE)
    test = test.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Further split test set into x and y
    X_test, y_test = tuple(zip(*test))
    X_test = np.squeeze(np.array(X_test))
    y_test = np.squeeze(np.array(y_test))

    return train, val, X_test, y_test


# Create data into training, validation, test sets WITH augmentation
def get_augmented_data():
    make_output_dir()
    if os.path.isdir(AUGMENT_DIR):
        print("Augmented image directory already exists. Please delete calling this function.")

    # Data augmentation parameters for training
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.2],
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generate augmented training images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=16,
        classes=input_files,
        class_mode='sparse',
        shuffle=True,
        seed=SEED,
        # not saving the images due to computer storage
    )

    # No augmentation for validation and test set
    val = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
        seed=SEED,
        batch_size=16,
    )    

    test = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=input_files,
        image_size=(224, 224),
        shuffle=True,
        batch_size=1,
        seed=SEED,
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    #train_generator = train_generator.cache().prefetch(buffer_size=AUTOTUNE)
    val = val.cache().prefetch(buffer_size=AUTOTUNE)
    test = test.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Further split test set into x and y
    X_test, y_test = tuple(zip(*test))
    X_test = np.squeeze(np.array(X_test))
    y_test = np.squeeze(np.array(y_test))

    return train_generator, val, X_test, y_test


# Returns a Tuple of Lists - (list of strings of class labels, list of ints corresponding to first list)
def get_class_labels():
    class_labels = input_files
    class_ints = [i for i in range(len(class_labels))]
    return class_labels, class_ints
