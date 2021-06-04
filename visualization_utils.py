"""Visualization code for saliency maps and class visualization.
	Source: https://github.com/timsainb/tensorflow-2-feature-visualization-notebooks
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
import tensorflow as tf
from tensorflow import keras
from tqdm.autonotebook import tqdm
import warnings

def plot_saliency_maps(model, plot_prefix, class_name):
	"""Plot saliency map for the first image in the dataset of 
	the given class.
	"""
	loaded_image = keras.preprocessing.image.load_img('Images/' + 
		class_name + '/' + class_name + '_1.jpeg',target_size=(224,224))
	# preprocess image to get it into the right format for the model
	image = keras.preprocessing.image.img_to_array(loaded_image)
	image = image.reshape((1, *image.shape))
	y_pred = model.predict(image)
	image_var = tf.Variable(image, dtype=float)

	with tf.GradientTape() as tape:
		pred = model(image_var, training=False)
		class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
		loss = pred[0][class_idxs_sorted[0]]
	grads = tape.gradient(loss, image_var)
	dgrad_abs = tf.math.abs(grads)
	dgrad_max = np.max(dgrad_abs, axis=3)[0]
	# normalize to range between 0 and 1
	arr_min, arr_max = np.min(dgrad_max), np.max(dgrad_max)
	grad_eval = (dgrad_max - arr_min) / (arr_max - arr_min + 1e-18)
	fig, axes = plt.subplots(1,2,figsize=(14,5))
	axes[0].imshow(loaded_image)
	axes[1].imshow(loaded_image)
	i = axes[1].imshow(grad_eval, cmap="jet", alpha=0.8) # , alpha=0.8
	colorbar = fig.colorbar(i)
	colorbar.set_label('Saliency', rotation=270)
	plt.title('Saliency map for ' + class_name + '_1')
	plt.tight_layout()
	plt.savefig(plot_prefix + '_' + class_name + '_1_saliency.png')
	plt.show()

def zero_one_norm(x):
	return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

def z_score(x, scale = 1.0):
	return (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x)/scale)

def norm(x):
	return zero_one_norm(z_score(x))

def soft_norm(x, n_std = 15):
	""" zscore and set n_std range of 0-1, then clip
	"""
	x = z_score(x) / (n_std*2)
	return tf.clip_by_value(x + 0.5, 0, 1)

def adjust_hsv(imgs, sat_exp = 2.0, val_exp = 0.5):
	""" normalize color for less emphasis on lower saturation
	"""
	# convert to hsv

	hsv = tf.image.rgb_to_hsv(imgs)
	hue, sat, val = tf.split(hsv, 3, axis=2)

	# manipulate saturation and value
	sat = tf.math.pow(sat,sat_exp)
	val = tf.math.pow(val,val_exp)
	# rejoin hsv
	hsv_new = tf.squeeze(tf.stack([hue, sat, val], axis=2), axis = 3)

	# convert to rgb
	rgb = tf.image.hsv_to_rgb(hsv_new)
	return rgb

def gen_noise(dim=224, nex = 1):
	""" Generate some noise to initialize
	"""
	input_img_data = tf.random.uniform((nex, dim, dim, 3))
	return tf.Variable(tf.cast(input_img_data, tf.float32))

def get_opt_function():
	""" This function returns the optimizer function. This is just necessary because of tensorflow weirdness. 
	See: https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-540071844
	"""
	@tf.function
	def opt(
		submodel,
		input_data,
		filter_index,
		optimizer,
		steps=100,
		lr=0.01,
		layer_dims=2,
		soft_norm_std=15,
		norm_f = "soft_norm",
		normalize_grads = True
	):
		""" This function runs a single optimization over the list of images
		"""
		# determine if this is a convolutional, or fully connected layer
		if layer_dims == 2:
			# identity function because second dimension is already the filter dimension
			loss_func = lambda out_: out_
		if layer_dims == 4:
			# flip to make filter dimension second dimension
			loss_func = lambda out_: tf.transpose(out_, perm = [0,3,1,2])
		# optimization
		for n in tf.range(steps):
			with tf.GradientTape() as tape:
				tape.watch(input_data)
				if norm_f == "sigmoid":
					outputs = submodel(tf.nn.sigmoid(input_data))
				else:
					outputs = submodel(input_data)
				outputs = loss_func(outputs)

				loss_value = tf.gather_nd(
					outputs, indices=filter_index, batch_dims=0, name=None
				)
				grads = tape.gradient(loss_value, input_data)

				if normalize_grads:
					norm_divisor = tf.expand_dims(
						tf.expand_dims(
							tf.expand_dims(tf.math.reduce_std(grads, axis=[1, 2, 3]), 1), 1
						),
						1,
					)
					normalized_grads = grads / norm_divisor
					optimizer.apply_gradients(zip([-normalized_grads], [input_data]))
				else: 
					optimizer.apply_gradients(zip([-grads], [input_data]))

				if norm_f == "clip":
					input_data.assign(tf.clip_by_value(input_data, 0, 1))
				elif norm_f == "soft_norm":
					input_data.assign(soft_norm(input_data, n_std=soft_norm_std))
				elif norm_f == "sigmoid":
					input_data.assign(tf.clip_by_value(input_data, -50, 50))
					# normalizes color channels to sit scale roughly uniform
					input_data.assign(z_score(input_data, scale=1.5))
		return input_data, loss_value
	return opt

def upscale_image(imgs, upscaling_factor=1.1, sigma=0):
	""" Upsample and smooth the list of images
	"""
	img_list = []
	for img in imgs:
		if upscaling_factor == 1.0:
			upscaled_img = img
		else:
			sz = np.array(np.shape(img))[0]
			sz_up = (upscaling_factor * sz).astype("int")
			lower = int(np.floor((sz_up - sz) / 2))
			upper = int(np.ceil((sz_up - sz) / 2))

			upscaled_img = resize(img.astype("float"), (sz_up, sz_up), anti_aliasing=True)
			upscaled_img = upscaled_img[
				lower:-lower, lower:-lower, :,
			]
		if sigma > 0:
			upscaled_img = gaussian(upscaled_img, sigma=sigma, multichannel=True)
		img_list.append(upscaled_img)
	return tf.Variable(tf.cast(np.array(img_list), tf.float32))

def optimize_filter(
	submodel,
	layer_name,
	filter_index,
	filters_shape,
	steps=20,
	lr=0.01,
	layer_dims=2,
	n_upsample=1,
	sigma=0,
	upscaling_factor=1.1,
	soft_norm_std=15,
	single_receptive_field=True,
	norm_f = "soft_norm",
	normalize_grads = True
):
	""" This pulls together the steps for optimizing the image, and upsampling
	"""
	warnings.filterwarnings('ignore')
	tf.autograph.set_verbosity(0)
	tf.get_logger().setLevel('ERROR')

	# opt = get_opt_function()
	optimizer = tf.keras.optimizers.Adam(lr)

	# subset center neuron if we only want to look at one receptive field
	if single_receptive_field & (layer_dims == 4):
		filter_index = [
			[i[0], i[1], int(filters_shape[1] / 2), int(filters_shape[2] / 2)]
			for i in filter_index
		]

	loss_list = []
	# list of outputs during optimization
	output_images = []
	# generate initial noise
	img_data = gen_noise(nex=len(filter_index))
	output_images.append(img_data.numpy())
	# apply optimization
	for i in tqdm(range(n_upsample), leave=False):
		opt = get_opt_function()
		# optimize
		img_data, loss = opt(
			submodel,
			img_data,
			filter_index,
			optimizer=optimizer,
			steps=steps,
			lr=lr,
			layer_dims=layer_dims,
			soft_norm_std=soft_norm_std,
			norm_f = norm_f,
			normalize_grads=normalize_grads
		)

		loss_list.append(np.mean(loss.numpy()))
		output_images.append(img_data.numpy())

		# upsample
		if i < (n_upsample - 1):
			img_data = upscale_image(
				img_data.numpy(), upscaling_factor=upscaling_factor, sigma=sigma,
			)

	# brg to rgb color channel conversion
	if norm_f == "sigmoid":
		output_images = [tf.nn.sigmoid(i).numpy()[:,:,:,::-1] for i in output_images]
	else:
		output_images = [i[:,:,:,::-1] for i in output_images]
	return output_images, loss_list

def get_class_index(classes, class_list):
	""" grabs the index in the predication layer of the network
		based on the class name
	"""
	filter_index = [np.where(class_list == i)[0][0] for i in classes]
	return filter_index

def display_features(output_images, plot_prefix, filter_titles=None, ncols=10, zoom = 5, sat_exp=2.0, val_exp = 1.0):
	"""Show the class visualizations for the input classes
	"""
	nrows = int(np.ceil(len(output_images[-1]) / ncols))
	fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*5,nrows*5))
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace = 0.1, wspace = 0.01)
	for axi, ax in enumerate(axs.flatten()):
		if filter_titles is not None:
			if axi < len(filter_titles):
				ax.set_title(filter_titles[axi], fontsize=20)
		ax.axis('off')

	for i in range(len(output_images[-1])):
		ax = axs.flatten()[i]
		rgb = adjust_hsv(output_images[-1][i], sat_exp=sat_exp, val_exp=val_exp)
		pt = ax.imshow(rgb)
	plt.savefig(plot_prefix + '_' + '_class_visualizations.png', bbox_inches='tight')
	plt.show()