from absl import app
from absl import flags
import tensorflow as tf
from tensorflow import keras
import os

import room_classifier
import data_generator


FLAGS = flags.FLAGS

flags.DEFINE_integer('model_id', None,
	'ID of the room classification model. If there is none (for example,'
	' if the model has not been made yet), one will be created.')

flags.DEFINE_integer('num_epochs', 10,
	'Number of epochs to train the data for.')

flags.DEFINE_float('learning_rate', 3e-4,
	'Learning rate for the classification model.')

flags.DEFINE_float('dropout_rate', 0.2,
	'Dropout rate for the classification model.')

flags.DEFINE_boolean('clean_images', False,
	'Whether or not to clean the input images.')

flags.DEFINE_boolean('finetune', True,
	'Whether or not to finetune the classification model.')


def main(argv):
	classifier = room_classifier.RoomClassifier(FLAGS.model_id, 
												FLAGS.learning_rate, 
												FLAGS.dropout_rate)
	print('Model ID: ', classifier.get_model_id())
	if FLAGS.clean_images:
		data_generator.clean_file_names()
	train_data, val_data, X_test, y_test = data_generator.get_data()
	if FLAGS.finetune:
		classifier.finetune(train_data, val_data, FLAGS.num_epochs)
		classifier.plot_model()
		classifier.export_model()
	class_labels, class_ints = data_generator.get_class_labels()
	classifier.evaluate(X_test, y_test, class_ints, class_labels)


if __name__ == '__main__':
	app.run(main)