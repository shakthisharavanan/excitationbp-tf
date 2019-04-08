"""
VGG-16 Imagenet classification using tensorflow slim
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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# def vgg_classifier(x):
# 	image_size = vgg.vgg_16.default_image_size
# 	with tf.Graph().as_default():
# 		normalized_image = vgg_preprocessing.preprocess_image(x, image_size, image_size, is_training=False)
# 		with slim.arg_scope(vgg.vgg_arg_scope()):
# 			output, _ = vgg.vgg_16(normalized_image, num_classes = 1000, is_training = False)
# 			probabilities = tf.nn.softmax(output)
# 		init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'), slim.get_model_variables('vgg_16'))
# 		sess = tf.Session()
# 		init_fn(sess)

# 	pass


if __name__ == "__main__":
	# Set some parameters
	image_size = vgg.vgg_16.default_image_size
	batch_size = 1

	image_file = "./data/imagenet/catdog/catdog.jpg"
	image = plt.imread(image_file)
	# plt.imshow(image)
	# plt.annotate('Something', xy = (0.05, 0.95), xycoords = 'axes fraction')
	# plt.show()

	labels = imagenet.create_readable_names_for_imagenet_labels()

	# Define graph
	with tf.Graph().as_default():
		x = tf.placeholder(dtype = tf.float32, shape = (image_size, image_size, 3))
		normalized_image = vgg_preprocessing.preprocess_image(x, image_size, image_size, is_training=False)
		normalized_images = tf.expand_dims(normalized_image, 0)
		with slim.arg_scope(vgg.vgg_arg_scope()):
			output, _ = vgg.vgg_16(normalized_images, num_classes = 1000, is_training = False)
			probabilities = tf.nn.softmax(output)
		init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'vgg_16.ckpt'), slim.get_model_variables('vgg_16'))
		# Run in a session
		with tf.Session() as sess:
			init_fn(sess)
			probability = sess.run(probabilities, feed_dict = {x: image})
			# pdb.set_trace()
			probability = probability[0, 0:]
			sorted_inds = [i[0] for i in sorted(enumerate(-probability), key=lambda x:x[1])]

			plt.imshow(image)
			for i in range(5):
				index = sorted_inds[i]
				print('Probability %0.2f%% => [%s]' % (probability[index] * 100, labels[index]))
			plt.annotate('{0}: {1: 0.2f}%'.format(labels[sorted_inds[0]], probability[sorted_inds[0]]*100), xy = (0.02, 0.95), xycoords = 'axes fraction')
			plt.show()

