"""
File to test vgg16 layers on slim
"""

import sys
import os
from time import sleep

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import pylab as plt
from skimage import transform
import scipy

slim_dir = "./models/research/slim/"
checkpoints_dir = "./checkpoints/"
sys.path.insert(0, slim_dir)
from nets import vgg
from preprocessing import vgg_preprocessing
from datasets import imagenet

from eb_fns.eb_fc import *
from eb_fns.eb_pool import *
from eb_fns.eb_conv import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CONTRASTIVE = True
save_index = 0

# Create a subdirectory to save the results
if not os.path.exists("output"):
    os.makedirs("output")


def plot_heatmap(image, excitation, title=None):
    global save_index
    (h, w, c) = image.shape
    heatmap = np.sum(excitation.squeeze(), axis=-1)
    # TODO: Activations look zoomed, need to find the reason
    heatmap_resized = transform.resize(heatmap.clip(min=0), (h, w), order=1, mode='constant')
    plt.imshow(image)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.7)
    if title:
        plt.title(title)

    # Save plot to the output directory
    plt.savefig('output/{}_{}.png'.format(save_index, title))
    save_index += 1
    plt.show()

if __name__ == "__main__":
    # Set some parameters
    image_size = vgg.vgg_16.default_image_size
    batch_size = 1

    # Uncomment one to take as sample input image
    # image_file = "./data/imagenet/catdog/catdog.jpg"
    # image_file = "./data/dome.jpg"
    # image_file = "./data/cat_1.jpg"
    # image_file = "./data/beer.jpg"
    image_file = "./data/elephant.jpeg"

    image = plt.imread(image_file)
    image = scipy.misc.imresize(image, (image_size, image_size))

    labels = imagenet.create_readable_names_for_imagenet_labels()

    # Dictionary to hold class indices
    class_idx = {}
    class_idx['Indian elephant, Elephas maximus'] = 385
    class_idx['tabby, tabby cat'] = 281


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
            probability = probability[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probability), key=lambda x: x[1])]

            for i in range(10):
                index = sorted_inds[i]
                print('Probability %0.2f%% => [%s]' % (probability[index] * 100, labels[index]))

            weights = tf.trainable_variables()
            weights_val = sess.run(weights)

            # Dictionary to store MWP
            P = {}

            # Set one hot vector for the winning class
            p = np.zeros((1, 1000))

            # Uncomment to run EB for most probable class
            # p[sorted_inds[0], 0] = 1

            # Else run EB for one of known classes
            p[0, class_idx['Indian elephant, Elephas maximus']] = 1       #
            P['fc8'] = np.copy(p)  # 1000 X 1

            """ For fc7 MWP """
            P['fc7'] = getMWPfc(P['fc8'].reshape(1,1,1,1000), weights_val[-2], layer_activations[-2])
            if CONTRASTIVE:
                cP = getMWPfc(P['fc8'].reshape(1, 1, 1, 1000), weights_val[-2] * -1, layer_activations[-2])
                P['fc7'] -= cP

            """ For fc6 MWP """
            P['fc6'] = getMWPfc(P['fc7'], weights_val[-4], layer_activations[-3])

            """ For pool5 MWP """
            P['pool5'] = getMWPfc(P['fc6'], weights_val[-6], layer_activations[-4])
            plot_heatmap(image, P['pool5'], "pool5")

            """ For conv5_3 MWP """
            # Get pool 5 to conv5_3 gradients
            dy_dx = sess.run(tf.gradients(endpoints['vgg_16/pool5'], endpoints['vgg_16/conv5/conv5_3']), feed_dict={x: image})[0]  # (1, 14, 14, 512)
            P['conv5_3'] = getMWPmaxpool(P['pool5'], layer_activations[-5], dy_dx)
            plot_heatmap(image, P['conv5_3'], "conv5_3")



            """ For conv5_2 MWP """
            P['conv5_2'] = getMWPconv(P['conv5_3'], weights_val[-8], layer_activations[-6])
            plot_heatmap(image, P['conv5_2'], "conv5_2")

            """ For conv5_1 MWP """
            P['conv5_1'] = getMWPconv(P['conv5_2'], weights_val[-10], layer_activations[-7])
            plot_heatmap(image, P['conv5_1'], "conv5_1")

            """ For pool4 MWP """
            P['pool4'] = getMWPconv(P['conv5_1'], weights_val[-12], layer_activations[-8])
            plot_heatmap(image, P['pool4'], "pool4")


            """ For conv4_3 MWP """
            # Get pool 4 to conv4_3 gradients
            dy_dx = sess.run(tf.gradients(endpoints['vgg_16/pool4'], endpoints['vgg_16/conv4/conv4_3']), feed_dict={x: image})[
                0]  # (1, 14, 14, 512)
            P['conv4_3'] = getMWPmaxpool(P['pool4'], layer_activations[-9], dy_dx)
            plot_heatmap(image, P['conv4_3'], "conv4_3")

            """ For conv4_2 MWP """
            P['conv4_2'] = getMWPconv(P['conv4_3'], weights_val[-14], layer_activations[-10])
            plot_heatmap(image, P['conv4_2'], "conv4_2")

            """ For conv4_1 MWP """
            P['conv4_1'] = getMWPconv(P['conv4_2'], weights_val[-16], layer_activations[-11])
            plot_heatmap(image, P['conv4_1'], "conv4_1")

            """ For pool3 MWP """
            P['pool3'] = getMWPconv(P['conv4_1'], weights_val[-18], layer_activations[-12])
            plot_heatmap(image, P['pool3'], "pool3")


            """ For conv3_3 MWP """
            # Get pool 3 to conv3_3 gradients
            dy_dx = \
            sess.run(tf.gradients(endpoints['vgg_16/pool3'], endpoints['vgg_16/conv3/conv3_3']), feed_dict={x: image})[
                0]  # (1, 14, 14, 512)
            P['conv3_3'] = getMWPmaxpool(P['pool3'], layer_activations[-13], dy_dx)
            plot_heatmap(image, P['conv3_3'], "conv3_3")

            """ For conv3_2 MWP """
            P['conv3_2'] = getMWPconv(P['conv3_3'], weights_val[-20], layer_activations[-14])
            plot_heatmap(image, P['conv3_2'], "conv3_2")

            """ For conv3_1 MWP """
            P['conv3_1'] = getMWPconv(P['conv3_2'], weights_val[-22], layer_activations[-15])
            plot_heatmap(image, P['conv3_1'], "conv3_1")

            """ For pool3 MWP """
            P['pool2'] = getMWPconv(P['conv3_1'], weights_val[-24], layer_activations[-16])
            plot_heatmap(image, P['pool2'], "pool2")


            """ For conv2_2 MWP """
            # Get pool 2 to conv2_2 gradients
            dy_dx = \
                sess.run(tf.gradients(endpoints['vgg_16/pool2'], endpoints['vgg_16/conv2/conv2_2']),
                         feed_dict={x: image})[
                    0]  # (1, 14, 14, 512)
            P['conv2_2'] = getMWPmaxpool(P['pool2'], layer_activations[-17], dy_dx)
            plot_heatmap(image, P['conv2_2'], "conv2_2")

            """ For conv2_1 MWP """
            P['conv2_1'] = getMWPconv(P['conv2_2'], weights_val[-26], layer_activations[-18])
            plot_heatmap(image, P['conv2_1'], "conv2_1")

            """ For conv2_1 MWP """
            P['pool1'] = getMWPconv(P['conv2_1'], weights_val[-28], layer_activations[-19])
            plot_heatmap(image, P['pool1'], "pool1")


            """ For conv1_2 MWP """
            # Get pool 1 to conv1_2 gradients
            dy_dx = \
                sess.run(tf.gradients(endpoints['vgg_16/pool1'], endpoints['vgg_16/conv1/conv1_2']),
                         feed_dict={x: image})[
                    0]  # (1, 14, 14, 512)
            P['conv1_2'] = getMWPmaxpool(P['pool1'], layer_activations[-20], dy_dx)
            plot_heatmap(image, P['conv1_2'], "conv1_2")

            """ For conv1_1 MWP """
            P['conv1_1'] = getMWPconv(P['conv1_2'], weights_val[-30], layer_activations[-21])
            plot_heatmap(image, P['conv1_1'], "conv1_1")

