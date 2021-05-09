import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd
import shutil
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import time
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.python.client import device_lib
# TODO: clean up order of imports

NUM_CLASSES = 67
IMG_SIZE = 224
BATCH_SIZE = 32

class RoomClassifier(object):
	"""Classify rooms based on input images.
	"""

	def __init__(self, model_id=None, lr=2e-3, dropout=0.2):
		self.setup()
		if model_id is None:
			self._model = self.create_model(lr, dropout)
			self._model_id = self.generate_model_id()
			self._model_path = self.format_model_path(self._model_id)
		else:
			self._model_id = model_id
			self._model_path = self.format_model_path(self._model_id)
			self._model = keras.load_model(self._model_path)

	def setup(self):
		"""Test whether system and GPU are configured correctly
		"""
		print(device_lib.list_local_devices())

	def create_model(self, lr, dropout):
		"""Load pretrained conv base model and set up for finetuning.

		Args:
			lr: learning rate for the model

		Returns:
			New keras model based on EfficientNet.
		"""
		# input_shape is (height, width, number of channels) for images
		input_shape = (IMG_SIZE, IMG_SIZE, 3) # 3?
		conv_base = EfficientNetB0(weights="imagenet", include_top=False, 
			input_shape=input_shape, drop_connect_rate=dropout)
		model = models.Sequential()
		model.add(conv_base)

		# rebuild top
		model.add(layers.GlobalMaxPooling2D(name="gap"))
		model.add(layers.BatchNormalization(name="batchnorm"))

		# avoid overfitting
		model.add(layers.Dropout(dropout, name="dropout"))
		model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="fc"))
		conv_base.trainable = False
		model.compile(
		    loss="categorical_crossentropy",
		    optimizer=optimizers.Adam(lr),
		    metrics=["acc"],
		)
		model.summary()
		return model

	def finetune(self, train_data, val_data, num_epochs):
		"""Finetune the model using the input data.

		Args:
			train_data: training data
			val_data: validation data
			num_epochs: number of epochs to train the data for

		Returns:
			Nothing, but the model is finetuned on the data.
		"""
		self._model.fit(
		    train_data,
		    #steps_per_epoch=train_data.n // BATCH_SIZE,
		    epochs=num_epochs,
		    validation_data=val_data,
		    #validation_steps=val_data.n // BATCH_SIZE,
		    verbose=1,
		    use_multiprocessing=True,
		    workers=4,
		)

	def evaluate(self, X_test, y_test):
		"""Evaluate the model on test data.
		"""
		y_pred = self._model.predict(X_test)
		score = self._model.evaluate(X_test, y_test, verbose=1)
		for name, value in zip(self._model.metrics_names, results):
			print(name, ": ", str(value))

		confusion = confusion_matrix(y_test, y_pred) 
		precision = precision_score(y_test, y_pred) 
		recall = recall_score(y_test, y_pred) 
		f1 = f1_score(y_test,y_pred) 

		fig, ax = plot_confusion_matrix(conf_mat=confusion,  figsize=(5, 5))
		plt.savefig('./plots/' + str(self._model_id) + '_confusion_matrix.png')
		plt.show()
		print("precision: ", precision)
		print("recall: ", recall)
		print("f1: ", f1)
		print(classification_report(y_test, y_pred))

	def plot_model(self):
		history = self._model.history

		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig('./plots/' + str(self._model_id) + '_scene_acc.png')
		plt.show()

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig('./plots/' + str(self._model_id) + '_scene_loss.png')
		plt.show()

	def generate_model_id(self):
		"""Generate a unique number for the current model.

		Returns:
			An integer uniquely identifying the model.
		"""
		t = time.time()
		return int(t)

	def get_model_id(self):
		return self._model_id

	def format_model_path(self, model_id):
		"""Generate the path to where the model is/can be saved.

		Args: 
			model_id: an integer uniquely identifying the model

		Returns:
			A string denoting the path where the model can be stored.
		"""
		model_path = './saved_models/scene_classifier_{}'.format(self._model_id) + '.h5'
		return model_path

	def get_model_path(self):
		return self._model_path

	def export_model(self):
		"""Saves the model to its corresponding path.
		"""
		self._model.save(self._model_path)
		print('Model successfully saved to ', self._model_path)