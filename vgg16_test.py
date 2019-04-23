"""
File to test vgg16 layers on slim
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def plot_heatmap(image, excitation):
    (h, w, c) = image.shape
    heatmap = np.sum(excitation, axis=2)
    heatmap_resized = transform.resize(heatmap, (h, w), order=3, mode='constant')
    plt.imshow(image)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    # Set some parameters
    image_size = vgg.vgg_16.default_image_size
    batch_size = 1

    image_file = "./data/imagenet/catdog/catdog.jpg"
    # image_file = "./data/dome.jpg"
    # image_file = "./data/cat_1.jpg"
    # image_file = "./data/beer.jpg"
    # image_file = "./data/elephant.jpeg"

    # image = Image.open(image_file)
    # image.thumbnail((image_size, image_size), Image.ANTIALIAS) # resizes image in-place
    image = plt.imread(image_file)
    # pdb.set_trace()
    image = scipy.misc.imresize(image, (image_size, image_size))

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
        # Run in a session
        with tf.Session() as sess:
            init_fn(sess)
            probability, layers = sess.run([probabilities, endpoints], feed_dict={x: image})
            layer_names, layer_activations = zip(*list(layers.items()))
            # pdb.set_trace()
            probability = probability[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probability), key=lambda x: x[1])]

            # pdb.set_trace()

            for i in range(10):
                index = sorted_inds[i]
                print('Probability %0.2f%% => [%s]' % (probability[index] * 100, labels[index]))
            # plt.imshow(image)
            # plt.annotate('{0}: {1: 0.2f}%'.format(labels[sorted_inds[0]], probability[sorted_inds[0]]*100), xy = (0.02, 0.95), xycoords = 'axes fraction')
            # plt.show()

            weights = tf.trainable_variables()
            weights_val = sess.run(weights)

            # Set MWP as a dict
            P = {}

            # Set one hot vector for the winning class
            p = np.zeros((1000, 1))
            p[sorted_inds[0], 0] = 1
            P['fc8'] = np.copy(p)  # 1000 X 1

            """ For fc7 MWP """
            # Get fc8 weights
            fc8_weights = np.copy((weights_val[-2])[0, 0])  # 4096 X 1000

            # Get fc7 activations
            fc7_activations = np.copy((layer_activations[-2])[0, 0]).T  # 4096 X 1

            # Calculate MWP of fc7 using Eq 10 in paper
            fc8_weights = fc8_weights.clip(min=0)  # threshold weights at 0
            m = np.dot(fc8_weights.T, fc7_activations)  # 1000 x 1
            n = P['fc8'] / m  # 1000 x 1
            o = np.dot(fc8_weights, n)  # 4096 x 1
            P['fc7'] = fc7_activations * o  # 4096 x 1

            """ For fc6 MWP """
            # Get fc7 weights
            fc7_weights = np.copy((weights_val[-4])[0, 0])  # 4096 X 4096

            # Get fc6 activations
            fc6_activations = np.copy((layer_activations[-3])[0, 0]).T  # 4096 X 1

            # Calculate MWP of fc6 using Eq 10 in paper
            fc7_weights = fc7_weights.clip(min=0)  # threshold weights at 0
            m = np.dot(fc7_weights.T, fc6_activations)  # 4096 x 1
            n = P['fc7'] / m  # 4096 x 1
            o = np.dot(fc7_weights, n)  # 4096 x 1
            P['fc6'] = fc6_activations * o  # 4096 * 1

            """ For pool5 MWP """
            # Get fc6 weights
            fc6_weights = np.copy(weights_val[-6])  # (7, 7, 512, 4096)
            fc6_weights_reshaped = fc6_weights.reshape(-1, 4096)  # (25088, 4096)

            # Get pool5 activations
            pool5_activations = np.copy(layer_activations[-4]).reshape(-1, 1)  # (25088, 1)

            # Calculate MWP of pool5 using Eq 10 in paper
            fc6_weights_reshaped = fc6_weights_reshaped.clip(min=0)  # threshold weights at 0
            m = np.dot(fc6_weights_reshaped.T, pool5_activations)  # 4096 x 1
            n = P['fc6'] / m  # 4096 x 1
            o = np.dot(fc6_weights_reshaped, n)  # 25088 x 1
            P['pool5'] = pool5_activations * o  # 25088 x 1
            P['pool5'] = P['pool5'].reshape(7, 7, 512)

            plot_heatmap(image, P['pool5'])

            """ For conv5_3 MWP """
            doubled_volume = P['pool5'].repeat(2, axis=0).repeat(2, axis=1)  # (14, 14, 512)
            # Get pool 5 to conv5_3 gradients
            dy_dx = sess.run(tf.gradients(endpoints['vgg_16/pool5'], endpoints['vgg_16/conv5/conv5_3']), feed_dict={x: image})[
                0][0]  # (14, 14, 512)
            P['conv5_3'] = dy_dx * doubled_volume

            plot_heatmap(image, P['conv5_3'])

            # Get conv5_3 weights
            conv5_3_weights= np.copy(weights_val[-8])  # (3, 3, 512, 512)

            # Get conv5_2 activations
            conv5_2_activations = np.copy(layer_activations[-6])  # (1, 14, 14, 512)
            print("Hello")

            x = np.transpose(conv5_2_activations, [0,3,1,2])
            cols = im2col_indices(x, 3, 3)  # (4608, 196)

            conv5_3_weights_reshaped = conv5_3_weights.reshape(-1, conv5_3_weights.shape[3]) # (4608, 512)

            # Calculate MWP of pool5 using Eq 10 in paper
            conv5_3_weights_reshaped = conv5_3_weights_reshaped.clip(min=0)  # threshold weights at 0
            m = np.dot(conv5_3_weights_reshaped.T, cols)  # 512 x 196
            n = P['conv5_3'].reshape(-1, 512).T / m  # 512 X 196
            o = np.dot(conv5_3_weights_reshaped, n)  # 4608 x 196
            k = cols * o  # 4608 x 196
            k = col2im_indices(k, x.shape, 3, 3)
            P['conv5_2'] = k.transpose([0, 2, 3, 1])

            plot_heatmap(image, P['conv5_2'][0])



            # heatmap = np.sum(P['pool5'], axis=2)
            # heatmap_resized = transform.resize(heatmap, (image_size, image_size), order=3, mode='constant')
            # plt.imshow(image)
            # plt.imshow(heatmap_resized, cmap='jet', alpha=0.7)
            # plt.show()

            # pdb.set_trace()

            sleep(0.1)
