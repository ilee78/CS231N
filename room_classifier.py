import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
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

NUM_CLASSES = 66
IMG_SIZE = 224
BATCH_SIZE = 32

class RoomClassifier(object):
	"""Classify rooms based on input images.
	"""

	def __init__(self, model_id=None, lr=3e-4, dropout=0.2):
		self.setup()
		self._unfrozen = False
		if model_id is None:
			self._model = self.create_model(lr, dropout)
			self._model_id = self.generate_model_id()
			self._model_path = self.format_model_path(self._model_id)
		else:
			self._model_id = model_id
			self._model_path = self.format_model_path(self._model_id)
			# Always loads models with EfficientNet frozen
			self._model = models.load_model(self._model_path)
		self._plot_prefix = self.generate_plot_prefix()

	def setup(self):
		"""Test whether system and GPU are configured correctly
		"""
		print(device_lib.list_local_devices())
		print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))


	def create_model(self, lr, dropout):
		"""Load pretrained conv base model and set up for finetuning.

		Args:
			lr: learning rate for the model

		Returns:
			New keras model based on EfficientNet.
		"""
		# input_shape is (height, width, number of channels) for images
		input_shape = (IMG_SIZE, IMG_SIZE, 3)
		conv_base = EfficientNetB2(weights="imagenet", include_top=False, 
			input_shape=input_shape) # , drop_connect_rate=dropout
		conv_base.trainable = False
		model = models.Sequential()
		model.add(conv_base)

		# Rescale inputs
		model.add(layers.experimental.preprocessing.Rescaling(1./IMG_SIZE, name="rescaling"))

		# rebuild top and add some dense layers
		# model.add(layers.Conv2D(32, (3,3), activation='relu'))
		# model.add(layers.BatchNormalization())
		model.add(layers.GlobalAveragePooling2D(name="gap"))
		model.add(layers.BatchNormalization(name="batchnorm"))
		# model.add(layers.Dropout(0.7, name="initial_dropout"))
		# model.add(layers.Dense(512, activation='relu', name="fc_512"))
		# model.add(layers.BatchNormalization())
		# model.add(layers.Activation('relu'))

		# avoid overfitting
		model.add(layers.Dropout(dropout, name="dropout"))
		model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="fc_output"))
		model.compile(
		    loss="sparse_categorical_crossentropy",
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
		# Save the model every epoch
		model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
		    filepath=self._model_path,
			save_weights_only=False,
			monitor='val_acc',
			mode='max',
			save_best_only=True,
			save_freq="epoch",
		)

		#reducing learning rate on plateau
		rlrop = keras.callbacks.ReduceLROnPlateau(
			monitor='val_loss', 
			mode='min', 
			patience= 5, 
			factor= 0.5, 
			min_lr= 1e-6, 
			verbose=1
		)

		self._model.fit(
			train_data,
			# steps_per_epoch=train_data.n // BATCH_SIZE,
			epochs=num_epochs,
			validation_data=val_data,
			# validation_steps=val_data.n // BATCH_SIZE,
			verbose=1,
			use_multiprocessing=True,
			workers=4,
			callbacks=[model_checkpoint_callback, rlrop],
		)

	def unfreeze(self, train_data, val_data, num_unfreeze, num_epochs, lr):
		"""Unfreeze the top num_layers_unfreeze layers while leaving BatchNorm layers frozen,
		then finetune the model again. Learning rate should be less than the initial
		finetuning learning rate.

		Args:
			train_data: training data
			val_data: validation data
			num_epochs: number of epochs to train the data for

		Returns:
			Nothing, but the model is finetuned on the data.
		"""
		self._unfrozen_layers = num_unfreeze
		self._model_path = self.format_model_path(self._model_id)
		self._plot_prefix = self.generate_plot_prefix()

		for layer in self._model.layers[-num_unfreeze:]:
			if not isinstance(layer, layers.BatchNormalization):
				layer.trainable = True
			else:
				layer.trainable = False
		self._model.compile(
			optimizer=optimizers.Adam(lr),
			loss="sparse_categorical_crossentropy",
			metrics=["acc"]
		)
		self._model.summary()
		self._model.fit(train_data, epochs=num_epochs, validation_data=val_data, verbose=1)

	def plot_model(self):
		history = self._model.history

		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(self._plot_prefix + '_acc.png')
		plt.show()

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(self._plot_prefix + '_loss.png')
		plt.show()

	def evaluate(self, X_test, y_test, class_ints, class_labels):
		"""Evaluate the model on test data.
		"""
		y_pred = self._model.predict(X_test)
		results = self._model.evaluate(X_test, y_test, verbose=1)
		for name, value in zip(self._model.metrics_names, results):
			print(name, ": ", str(value))

		y_pred = np.argmax(y_pred, axis=1)
		confusion = confusion_matrix(y_test, y_pred, normalize='true') 
		precision = precision_score(y_test, y_pred, average='micro') 
		recall = recall_score(y_test, y_pred, average='micro') 
		f1 = f1_score(y_test,y_pred, average='micro') 

		# fig, ax = plot_confusion_matrix(conf_mat=confusion, 
		# 								figsize=(30, 30),
		# 								show_absolute=False,
		# 								show_normed=True,
		# 								colorbar=True)
		fig, ax = plt.subplots(figsize=(16, 13))
		ax = sns.heatmap(confusion,
						vmin=0, vmax=1,
						annot=False, cmap='Blues', cbar=True,
						xticklabels=class_labels, yticklabels=class_labels)
		plt.tight_layout()
		plt.savefig(self._plot_prefix + '_confusion_matrix.png')
		plt.show()
		print("precision: ", precision)
		print("recall: ", recall)
		print("f1: ", f1)
		print(classification_report(y_test, y_pred, labels=class_ints, target_names=class_labels))

	def generate_model_id(self):
		"""Generate a unique number for the current model.

		Returns:
			An integer uniquely identifying the model.
		"""
		t = time.time()
		return int(t)

	def get_model_id(self):
		"""Returns id of the model
		"""
		return self._model_id

	def generate_plot_prefix(self):
		"""Generates prefix to path of plots. Dependent on model id and whether
		or not the model has been unfrozen.
		"""
		prefix = './plots/' + str(self._model_id)
		if self._unfrozen:
			prefix += '_' + self._num_unfrozen + 'unfrozen'
		return prefix

	def format_model_path(self, model_id):
		"""Generate the path to where the model is/can be saved.

		Args: 
			model_id: an integer uniquely identifying the model

		Returns:
			A string denoting the path where the model can be stored.
		"""
		model_path = './saved_models/scene_classifier_{}'.format(self._model_id)
		if self._unfrozen:
			model_path += '_' + self._num_unfrozen
		model_path += '.h5'
		return model_path

	def get_model_path(self):
		return self._model_path

	def export_model(self):
		"""Saves the model to its corresponding path.
		"""
		self._model.save(self._model_path)
		print('Model successfully saved to ', self._model_path)
