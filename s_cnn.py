import tensorflow as tf 
import numpy as np
import random

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCH = 2

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def inf(x):
    w_conv1 = tf.get_variable('w_conv1', [5, 5, 1, 24])
    b_conv1 = tf.get_variable('b_conv1', [24])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    
    h_pool1 = max_pool_2x2(h_conv1)
    
    w_conv2 = tf.get_variable('w_conv2', [3, 3, 24, 48])
    b_conv2 = tf.get_variable('b_conv2', [48])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    
    h_pool2 = max_pool_2x2(h_conv2)
    
    w_conv3 = tf.get_variable('w_conv3', [3, 3, 48, 24])
    b_conv3 = tf.get_variable('b_conv3', [24])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    
    w_conv4 = tf.get_variable('w_conv4', [3, 3, 24, 12])
    b_conv4 = tf.get_variable('b_conv4', [12])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, w_conv4) + b_conv4)
    
    return h_conv4

