import tensorflow as tf
import cv2
import numpy as np
import os
import math
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

START_NUM = 1
MAX_NUM = 10
IMG_PATH = './shanghaitech/part_A_final/test_data/images/'

class net:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_pre = self.inf(self.x)
    
    def final(self, sess):
        for img_num in range(START_NUM, MAX_NUM + 1):
            img = cv2.imread(IMG_PATH + 'IMG_' + str(img_num) + '.jpg', 0)
            height = img.shape[0]
            width = img.shape[1]
            
            x_feed = np.array(img * 1.0 / 255.0, dtype = 'float32')
            x_feed = np.reshape(x_feed, (1, height, width, 1))
            
            print(img_num, '/ ', MAX_NUM)

            den = sess.run(self.y_pre, feed_dict = {self.x: x_feed})    
            den = np.reshape(den, (den.shape[1], den.shape[2]))
            ##绘图部分
            
            max = float(np.max(den))
            den = den * 255 / max
        
            x = []
            y = []
            for i in range(len(den)):
                for j in range(len(den[i])):
                    for k in range(int(den[i][j])):
                        x.append(j)
                        y.append(len(den) - i)
            for i in range(len(den)):
                for j in range(len(den[i])):
                    x.append(j)
                    y.append(len(den) - i)
        
            counts, xbins, ybins, image = plt.hist2d(x, y, bins = 100, norm = LogNorm(), cmap = plt.cm.rainbow)
            plt.savefig('density_' + str(img_num) + '.png')
            #plt.show()
            
        
    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    def inf(self, x):
        w_conv1 = tf.get_variable('w_conv1', [5, 5, 1, 24])
        b_conv1 = tf.get_variable('b_conv1', [24])
        h_conv1 = tf.nn.relu(self.conv2d(x, w_conv1) + b_conv1)
        
        h_pool1 = self.max_pool_2x2(h_conv1)
        
        w_conv2 = tf.get_variable('w_conv2', [3, 3, 24, 48])
        b_conv2 = tf.get_variable('b_conv2', [48])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        
        h_pool2 = self.max_pool_2x2(h_conv2)
        
        w_conv3 = tf.get_variable('w_conv3', [3, 3, 48, 24])
        b_conv3 = tf.get_variable('b_conv3', [24])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, w_conv3) + b_conv3)
        
        w_conv4 = tf.get_variable('w_conv4', [3, 3, 24, 12])
        b_conv4 = tf.get_variable('b_conv4', [12])
        h_conv4 = tf.nn.relu(self.conv2d(h_conv3, w_conv4) + b_conv4)
        
        w_conv5 = tf.get_variable('w_conv5', [1, 1, 12, 1])
        b_conv5 = tf.get_variable('b_conv5', [1])
        h_conv5 = tf.nn.relu(self.conv2d(h_conv4, w_conv5) + b_conv5)
        
        y_pre = h_conv5
        
        return y_pre
        
a = net()
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    a.final(sess)