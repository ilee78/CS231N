from absl import app
from absl import flags
import os
import tensorflow as tf
from tensorflow import keras

import room_classifier
import data_generator


FLAGS = flags.FLAGS

flags.DEFINE_integer('model_id', None,
	'ID of the room classification model. If there is none (for example,'
	' if the model has not been made yet), one will be created.')

flags.DEFINE_integer('num_epochs', 20,
	'Number of epochs to train the data for.')

flags.DEFINE_integer('unfreeze', 0,
	'How many layers of the model to unfreeze, to further finetune. '
	'This operation only works if finetune is False.')

flags.DEFINE_float('learning_rate', 1e-3,
	'Learning rate for the classification model.')

flags.DEFINE_float('dropout_rate', 0.5,
	'Dropout rate for the classification model.')

flags.DEFINE_boolean('clean_images', False,
	'Whether or not to clean the input images.')

flags.DEFINE_boolean('finetune', True,
	'Whether or not to finetune the classification model.')


def main(argv):
	# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	# with tf.device(tf.DeviceSpec(device_type='GPU', device_index=0)):
	classifier = room_classifier.RoomClassifier(FLAGS.model_id, 
										FLAGS.learning_rate, 
										FLAGS.dropout_rate)
	print('Model ID: ', classifier.get_model_id())
	if FLAGS.clean_images:
		data_generator.clean_file_names()
	train_data, val_data, X_test, y_test = data_generator.get_data()
	if FLAGS.finetune:
		classifier.finetune(train_data, val_data, FLAGS.num_epochs)
		classifier.plot_history()
		# classifier.export_model()
	elif FLAGS.unfreeze > 0:
		classifier.unfreeze(train_data, val_data, FLAGS.unfreeze,
			FLAGS.num_epochs, FLAGS.learning_rate)
		classifier.plot_history()
		classifier.export_model()
	classifier.plot_saliency_visualization(['bathroom', 'concert_hall', 'closet', 'poolinside'])
	class_labels, class_ints = data_generator.get_class_labels()
	classifier.evaluate(X_test, y_test, class_ints, class_labels)


if __name__ == '__main__':
	app.run(main)
