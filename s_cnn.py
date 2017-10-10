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
    