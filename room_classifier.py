import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from numpy.random import seed
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
from tensorflow.keras import models, layers, optimizers
from tensorflow.python.client import device_lib

import data_generator
import visualization_utils

NUM_CLASSES = 66
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 1234

seed(SEED) # keras seed fixing
tf.random.set_seed(SEED)# tensorflow seed fixing

class RoomClassifier(object):
	"""Classify rooms based on input images.
	"""

	def __init__(self, model_id=None, lr=3e-4, dropout=0.2):
		self._setup()
		self._unfrozen = False
		if model_id is None:
			self._model = self._create_model(lr, dropout)
			self._model_id = self._generate_model_id()
			self._model_path = self._format_model_path(self._model_id)
		else:
			self._model_id = model_id
			self._model_path = self._format_model_path(self._model_id)
			# Always loads models with EfficientNet frozen
			self._model = models.load_model(self._model_path)
			self._model.load_weights(self._model_path)
			# self._model.compile(
			# 	loss="sparse_categorical_crossentropy",
			#     optimizer=optimizers.Adam(lr),
			#     metrics=['sparse_categorical_accuracy'])
			self._model.summary()
		self._plot_prefix = self._generate_plot_prefix()

	def _setup(self):
		"""Test whether system and GPU are configured correctly
		"""
		print(device_lib.list_local_devices())
		print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))


	def _create_model(self, lr, dropout):
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
		# model.add(layers.Conv2D(64, (3,3), activation='relu'))
		#model.add(layers.BatchNormalization())
		# model.add(layers.MaxPool2D((2, 2), 2, padding='same', name="max_pool"))
		#model.add(layers.Conv2D(128, (3,3), activation='relu'))
		model.add(layers.GlobalAveragePooling2D(name="gap"))
		model.add(layers.BatchNormalization(name="batchnorm"))
		model.add(layers.Dropout(0.3, name="fixed_dropout"))
		# model.add(layers.Dense(64, activation='relu', name="fc_64"))
		# model.add(layers.Dropout(0.5, name="second_dropout"))
		model.add(layers.Dense(512, activation='relu', name="fc_512"))
		# model.add(layers.Dense(128, activation='relu', name="fc_128"))
		# model.add(layers.BatchNormalization(name="batchnorm_2"))
		# model.add(layers.Activation('relu'))

		# avoid overfitting
		model.add(layers.Dropout(dropout, name="dropout")) 
		model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="predictions"))
		model.compile(
		    loss="sparse_categorical_crossentropy",
		    optimizer=optimizers.Adam(lr),
		    metrics=['sparse_categorical_accuracy'], # tf.keras.metrics.SparseCategoricalAccuracy(dtype=tf.float32)
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
			monitor='val_sparse_categorical_accuracy',
			mode='max',
			save_best_only=True,
			save_freq="epoch",
			verbose=0
		)

		# Reduce learning rate on plateau
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
		self._model_path = self._format_model_path(self._model_id)
		self._plot_prefix = self._generate_plot_prefix()

		for layer in self._model.efficientnetb0.layers[-num_unfreeze:]:
			if not isinstance(layer, layers.BatchNormalization):
				layer.trainable = True
			else:
				layer.trainable = False
		self._model.compile(
			optimizer=optimizers.Adam(lr),
			loss="sparse_categorical_crossentropy",
			metrics=["sparse_categorical_accuracy"]
		)
		self._model.summary()
		self._model.fit(train_data, epochs=num_epochs, validation_data=val_data, verbose=1)

	def plot_history(self):
		history = self._model.history

		plt.plot(history.history['sparse_categorical_accuracy'])
		plt.plot(history.history['val_sparse_categorical_accuracy'])
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

	def plot_saliency_visualization(self, class_names):
		"""Plot saliency maps for the first image in the given classes
		as well as class visualizations for those classes.

		Args:
			class_name: list containing the name of the classes

		Returns:
			Nothing, but displays and saves the requested plots.
		"""
		# Saliency map first
		# for class_name in class_names:
		# 	visualization_utils.plot_saliency_maps(self._model, self._plot_prefix, class_name)

		# Class visualization
		class_labels, class_ints = data_generator.get_class_labels()
		classes = pd.DataFrame({
			'label': class_ints,
			'name': class_labels
			})
		class_list = classes.name.values
		filter_index = visualization_utils.get_class_index(class_names, class_list)
		layer_name = "fc" # CHANGE for each model
		filter_index = [[i, j] for i, j in enumerate(filter_index)]
		# get module of input/output
		submodel = tf.keras.models.Model(
			[self._model.inputs[0]], [self._model.get_layer(layer_name).output]
		)
		filters_shape = submodel.outputs[0].shape
		output_images, loss_list = visualization_utils.optimize_filter(
			submodel,
			layer_name,
			filter_index,
			filters_shape=filters_shape,
			# steps = 20, # how many training steps to perform
			# lr=0.1, # gradient step size 
			layer_dims=len(submodel.outputs[0].shape), # how many dimensions the output layer is (2 for fully connected, 4 for convolutional)
			n_upsample=50, # how many steps to upsample
			sigma=1.0, # the amount of blurring to perform when upsampling
			upscaling_factor=1.01, # how much to upsample by
			single_receptive_field=False, # whether to optimize a single neuron, or optimize over the layer
			norm_f="sigmoid", # how to normalize/color channels between 0 and 1 (clip, sigmoid, )
			soft_norm_std = 3, # the number of standard deviations to clip if norm_f is soft_norm (lower = more saturated)
			normalize_grads=True
		)
		visualization_utils.display_features(output_images, self._plot_prefix, class_names, ncols=4, zoom=5)

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

	def _generate_model_id(self):
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

	def _generate_plot_prefix(self):
		"""Generates prefix to path of plots. Dependent on model id and whether
		or not the model has been unfrozen.
		"""
		prefix = './plots/' + str(self._model_id)
		if self._unfrozen:
			prefix += '_' + self._num_unfrozen + 'unfrozen'
		return prefix

	def _format_model_path(self, model_id):
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
