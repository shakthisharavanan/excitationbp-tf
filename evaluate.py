"""
Code to evaluate EB on Imagenet bounding box
"""

import sys
import os
from os import path
import time
from time import sleep

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import pylab as plt

from tqdm import tqdm, trange, tqdm_notebook, tnrange
import glob
import time
import pdb

slim_dir = "./models/research/slim/"
checkpoints_dir = "./checkpoints/"
sys.path.insert(0, slim_dir)
from nets import vgg
from preprocessing import vgg_preprocessing
from datasets import imagenet
from skimage import transform, filters
from PIL import Image
import scipy

from im2col import *

from eb_fns.eb_fc import *
from eb_fns.eb_pool import *
from eb_fns.eb_conv import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def plot_heatmap(image, excitation, title=None):
	(h, w, c) = image.shape
	heatmap = np.sum(excitation.squeeze(), axis=-1)
	# heatmap_resized = scipy.misc.imresize(heatmap, (h, w)).clip(min = 0)
	# heatmap_resized = np.pad(heatmap_resized, ((1,1), (1,1)), mode = 'constant', constant_values = 0)
	heatmap_resized = transform.resize(heatmap, (h, w), order=3, mode='constant').clip(min=0)
	# heatmap_resized = np.pad(heatmap_resized, ((1,1), (1,1)), mode = 'constant', constant_values = 0)
	# heatmap_resized = transform.resize(heatmap_resized, (h, w), order=3, mode='constant').clip(min=0)
	# pdb.set_trace()
	# print(heatmap_resized)
	plt.imshow(image)
	# y,x = (heatmap_resized.argmax()//h, heatmap_resized.argmax() - heatmap_resized.argmax()//h * h)
	y,x = np.unravel_index(heatmap_resized.argmax(), heatmap_resized.shape)
	print(heatmap_resized.argmax(), x, y)
	plt.plot(x, y, "*", color = 'green')
	plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
	if title:
		plt.title(title)
	plt.show()

def find_argmax(image, excitation, title=None):
	(h, w, c) = image.shape
	heatmap = np.sum(excitation.squeeze(), axis=-1)
	# heatmap_resized = scipy.misc.imresize(heatmap, (h, w)).clip(min = 0)
	heatmap_resized = transform.resize(heatmap, (h, w), order=3, mode='constant').clip(min=0)
	y,x = np.unravel_index(heatmap_resized.argmax(), heatmap_resized.shape)
	return x, y

if __name__ == "__main__":
	# Set some parameters
	image_size = vgg.vgg_16.default_image_size
	batch_size = 1


	image_paths = sorted(glob.glob("/mnt/workspace/datasets/ImageNet/tabby/filtered/filtered_images/*"))
	eb_points_path = "./eb_points_path/"

	# image_file = "./data/imagenet/catdog/catdog.jpg"
	# image_file = "./data/dome.jpg"
	# image_file = "./data/cat_1.jpg"
	# image_file = "./data/beer.jpg"
	# image_file = "./data/elephant.jpeg"

	# image = Image.open(image_file)
	# image.thumbnail((image_size, image_size), Image.ANTIALIAS) # resizes image in-place
	# image = plt.imread(image_file)
	# pdb.set_trace()
	# image_resized = scipy.misc.imresize(image, (image_size, image_size))

	# image = transform.resize(image, (image_size, image_size))
	# pdb.set_trace()
	# plt.imshow(image)
	# plt.annotate('Something', xy = (0.05, 0.95), xycoords = 'axes fraction')
	# plt.show()

	labels = imagenet.create_readable_names_for_imagenet_labels()

	# Define graph
	with tf.Graph().as_default():
		x = tf.placeholder(dtype=tf.float32, shape=(image_size, image_size, 3))
		normalized_image = vgg_preprocessing.preprocess_image(x, image_size, image_size, is_training=False)
		normalized_images = tf.expand_dims(normalized_image, 0)
		with slim.arg_scope(vgg.vgg_arg_scope()):
			output, endpoints = vgg.vgg_16(normalized_images, num_classes=1000, is_training=False)
			probabilities = tf.nn.softmax(output)
		init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
												 slim.get_model_variables('vgg_16'))


		with tf.Session() as sess:
			# Run in a session
			init_fn(sess)
			with tqdm(total = len(image_paths)) as outer_pbar:
				for image_file in image_paths:

					outer_pbar.set_description("Total Progress")
					outer_pbar.update(1)

					image = plt.imread(image_file)

					if len(image.shape)<3:
						print(image_file)
						continue
					(h, w, c) = image.shape
					image_resized = scipy.misc.imresize(image, (image_size, image_size))

					numpy_path = eb_points_path + image_file.split('/')[-1].split('.')[0]
					output_array = np.zeros((2,2,2))
					
					probability, layers = sess.run([probabilities, endpoints], feed_dict={x: image_resized})
					layer_names, layer_activations = zip(*list(layers.items()))
					probability = probability[0, 0:]
					# sorted_inds = [i[0] for i in sorted(enumerate(-probability), key=lambda x: x[1])]

					# pdb.set_trace()

					# for i in range(10):
					# 	index = sorted_inds[i]
					# 	print('Probability %0.2f%% => [%s]' % (probability[index] * 100, labels[index]))
					# plt.imshow(image)
					# plt.annotate('{0}: {1: 0.2f}%'.format(labels[sorted_inds[0]], probability[sorted_inds[0]]*100), xy = (0.02, 0.95), xycoords = 'axes fraction')
					# plt.show()

					weights = tf.trainable_variables()
					weights_val = sess.run(weights)

					# Set MWP as a dict
					P = {}
					cP = {}

					# Set one hot vector for the winning class
					p = np.zeros((1000, 1))
					# p[sorted_inds[0], 0] = 1
					p[281, 0] = 1
					P['fc8'] = np.copy(p)  # 1000 X 1
					cP['fc8'] = np.copy(p)

					P['fc7'] = getMWPfc(P['fc8'].T.reshape(1,1,1,1000), weights_val[-2], layer_activations[-2])
					c_mwp = getMWPfc(P['fc8'].T.reshape(1,1,1,1000), weights_val[-2] * -1, layer_activations[-2])
					cP['fc7'] = P['fc7'] - c_mwp

					P['fc6'] = getMWPfc(P['fc7'], weights_val[-4], layer_activations[-3])
					cP['fc6'] = getMWPfc(cP['fc7'], weights_val[-4], layer_activations[-3])

					P['pool5'] = getMWPfc(P['fc6'], weights_val[-6], layer_activations[-4])
					cP['pool5'] = getMWPfc(cP['fc6'], weights_val[-6], layer_activations[-4])

					xmax, ymax = find_argmax(image, P['pool5'], "pool5")
					output_array[0, 0, 0] = xmax
					output_array[0, 0, 1] = ymax
					xmax, ymax = find_argmax(image, cP['pool5'], "pool5")
					output_array[1, 0, 0] = xmax
					output_array[1, 0, 1] = ymax

					# pdb.set_trace()
					# plot_heatmap(image, P['pool5'], "pool5")

					# Get pool 5 to conv5_3 gradients
					dy_dx = sess.run(tf.gradients(endpoints['vgg_16/pool5'], endpoints['vgg_16/conv5/conv5_3']), feed_dict={x: image_resized})[0]  # (1, 14, 14, 512)
					P['conv5_3'] = getMWPmaxpool(P['pool5'], layer_activations[-5], dy_dx)
					cP['conv5_3'] = getMWPmaxpool(cP['pool5'], layer_activations[-5], dy_dx)

					# plot_heatmap(image, P['conv5_3'], "conv5_3")


					P['conv5_2'] = getMWPconv(P['conv5_3'], weights_val[-8], layer_activations[-6])
					cP['conv5_2'] = getMWPconv(cP['conv5_3'], weights_val[-8], layer_activations[-6])

					# plot_heatmap(image, P['conv5_2'], "conv5_2")

					""" For conv5_1 MWP """
					P['conv5_1'] = getMWPconv(P['conv5_2'], weights_val[-10], layer_activations[-7])
					cP['conv5_1'] = getMWPconv(cP['conv5_2'], weights_val[-10], layer_activations[-7])

					# plot_heatmap(image, P['conv5_1'], "conv5_1")

					""" For pool4 MWP """
					P['pool4'] = getMWPconv(P['conv5_1'], weights_val[-12], layer_activations[-8])
					cP['pool4'] = getMWPconv(cP['conv5_1'], weights_val[-12], layer_activations[-8])


					# y,x = (heatmap_resized.argmax()//h, heatmap_resized.argmax() - heatmap_resized.argmax()//h * h)

					# plot_heatmap(image, P['pool4'], "pool4")
					# plot_heatmap(image, cP['pool4'], "pool4")

					xmax, ymax = find_argmax(image, P['pool4'], "pool4")
					output_array[0, 1, 0] = xmax
					output_array[0, 1, 1] = ymax
					xmax, ymax = find_argmax(image, cP['pool4'], "pool4")
					output_array[1, 1, 0] = xmax
					output_array[1, 1, 1] = ymax

					np.save(numpy_path, output_array)
					# pdb.set_trace()


					# """ For conv4_3 MWP """
					# # Get pool 5 to conv5_3 gradients
					# dy_dx = sess.run(tf.gradients(endpoints['vgg_16/pool4'], endpoints['vgg_16/conv4/conv4_3']), feed_dict={x: image})[
					# 	0]  # (1, 14, 14, 512)
					# P['conv4_3'] = getMWPmaxpool(P['pool4'], layer_activations[-9], dy_dx)

					# plot_heatmap(image, P['conv4_3'], "conv4_3")

					# """ For conv4_2 MWP """
					# P['conv4_2'] = getMWPconv(P['conv4_3'], weights_val[-14], layer_activations[-10])

					# plot_heatmap(image, P['conv4_2'], "conv4_2")

					# """ For conv4_1 MWP """
					# P['conv4_1'] = getMWPconv(P['conv4_2'], weights_val[-16], layer_activations[-11])

					# plot_heatmap(image, P['conv4_1'], "conv4_1")

					# """ For pool3 MWP """
					# P['pool3'] = getMWPconv(P['conv4_1'], weights_val[-18], layer_activations[-12])

					# plot_heatmap(image, P['pool3'], "pool3")



					sleep(0.1)