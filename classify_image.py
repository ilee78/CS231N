"""Given the filename of an image, classify the image using our model.
"""

from absl import app, flags
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import room_classifier
import data_generator

BEST_MODEL_ID = 1622448785
IMG_SIZE = 224
FLAGS = flags.FLAGS

flags.DEFINE_string('image_name', 'Images/bar/bar_1.jpeg',
	'Name of the input image that is to be classified')

flags.DEFINE_integer('model_id', BEST_MODEL_ID,
	'ID of the finetuned model to use.')

def main(argv):
	class_labels, class_ints = data_generator.get_class_labels()

	# Set up model
	model_path = './saved_models/scene_classifier_{}.h5'.format(FLAGS.model_id)
	model = models.load_model(model_path)
	model.load_weights(model_path)
	print('Loaded model', FLAGS.model_id)

	# Keep going until the user wants to stop
	while True:
		# Prompt user for image name
		image_name = input('Please enter the name of an image! (press ENTER to quit) ')
		if image_name == '':
			break

		if os.path.isfile(image_name):
			# Set up image
			image = tf.keras.preprocessing.image.load_img(image_name, target_size=(224, 224))
			input_arr = keras.preprocessing.image.img_to_array(image)

			predictions = model(input_arr[tf.newaxis,...])
			top5 = np.array(class_labels)[np.argsort(predictions)[0,::-1][:5]]
			print("Top 5 predictions:\n", top5)
		else:
			print('Invalid image name!')


if __name__ == '__main__':
	app.run(main)