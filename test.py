import numpy as np 
import cv2
import tensorflow as tf
import os
import random
import math

LEARNING_RATE = 1e-4

dataset = 'A'

img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
den_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'

def data_pre():
    print('loading data...')
    img_names = os.listdir(img_path)
    img_num = len(img_names)

    data = []
    for i in range(img_num):
        if i % 100 == 0:
            print(i, '/', img_num)
        name = img_names[i]
        #print(name + '****************************')
        img = cv2.imread(img_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        #print(img.shape)
        den = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
        #print(den_quarter.shape)
        den_sum = np.sum(den)
        data.append([img, den_sum])

    print('load data finished.')
    return data

class net:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_act = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_pre = self.inf(self.x)

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.y_act - self.y_pre)))
        self.act_sum = tf.reduce_sum(self.y_act)
        self.pre_sum = tf.reduce_sum(self.y_pre)
        self.MAE = tf.abs(self.act_sum - self.pre_sum)

        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, 'model' + dataset + '/model.ckpt')
            self.test(sess)

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    def inf(self, x):
        # s net ###########################################################
        w_conv1_1 = tf.get_variable('w_conv1_1', [5, 5, 1, 24])
        b_conv1_1 = tf.get_variable('b_conv1_1', [24])
        h_conv1_1 = tf.nn.relu(self.conv2d(x, w_conv1_1) + b_conv1_1)

        h_pool1_1 = self.max_pool_2x2(h_conv1_1)

        w_conv2_1 = tf.get_variable('w_conv2_1', [3, 3, 24, 48])
        b_conv2_1 = tf.get_variable('b_conv2_1', [48])
        h_conv2_1 = tf.nn.relu(self.conv2d(h_pool1_1, w_conv2_1) + b_conv2_1)

        h_pool2_1 = self.max_pool_2x2(h_conv2_1)

        w_conv3_1 = tf.get_variable('w_conv3_1', [3, 3, 48, 24])
        b_conv3_1 = tf.get_variable('b_conv3_1', [24])
        h_conv3_1 = tf.nn.relu(self.conv2d(h_pool2_1, w_conv3_1) + b_conv3_1)

        w_conv4_1 = tf.get_variable('w_conv4_1', [3, 3, 24, 12])
        b_conv4_1 = tf.get_variable('b_conv4_1', [12])
        h_conv4_1 = tf.nn.relu(self.conv2d(h_conv3_1, w_conv4_1) + b_conv4_1)
        
        # m net ###########################################################
        w_conv1_2 = tf.get_variable('w_conv1_2', [7, 7, 1, 20])
        b_conv1_2 = tf.get_variable('b_conv1_2', [20])
        h_conv1_2 = tf.nn.relu(self.conv2d(x, w_conv1_2) + b_conv1_2)

        h_pool1_2 = self.max_pool_2x2(h_conv1_2)

        w_conv2_2 = tf.get_variable('w_conv2_2', [5, 5, 20, 40])
        b_conv2_2 = tf.get_variable('b_conv2_2', [40])
        h_conv2_2 = tf.nn.relu(self.conv2d(h_pool1_2, w_conv2_2) + b_conv2_2)

        h_pool2_2 = self.max_pool_2x2(h_conv2_2)

        w_conv3_2 = tf.get_variable('w_conv3_2', [5, 5, 40, 20])
        b_conv3_2 = tf.get_variable('b_conv3_2', [20])
        h_conv3_2 = tf.nn.relu(self.conv2d(h_pool2_2, w_conv3_2) + b_conv3_2)

        w_conv4_2 = tf.get_variable('w_conv4_2', [5, 5, 20, 10])
        b_conv4_2 = tf.get_variable('b_conv4_2', [10])
        h_conv4_2 = tf.nn.relu(self.conv2d(h_conv3_2, w_conv4_2) + b_conv4_2)
        
        #l net ###########################################################
        w_conv1_3 = tf.get_variable('w_conv1_3', [9, 9, 1, 16])
        b_conv1_3 = tf.get_variable('b_conv1_3', [16])
        h_conv1_3 = tf.nn.relu(self.conv2d(x, w_conv1_3) + b_conv1_3)

        h_pool1_3 = self.max_pool_2x2(h_conv1_3)

        w_conv2_3 = tf.get_variable('w_conv2_3', [7, 7, 16, 32])
        b_conv2_3 = tf.get_variable('b_conv2_3', [32])
        h_conv2_3 = tf.nn.relu(self.conv2d(h_pool1_3, w_conv2_3) + b_conv2_3)

        h_pool2_3 = self.max_pool_2x2(h_conv2_3)

        w_conv3_3 = tf.get_variable('w_conv3_3', [7, 7, 32, 16])
        b_conv3_3 = tf.get_variable('b_conv3_3', [16])
        h_conv3_3 = tf.nn.relu(self.conv2d(h_pool2_3, w_conv3_3) + b_conv3_3)

        w_conv4_3 = tf.get_variable('w_conv4_3', [7, 7, 16, 8])
        b_conv4_3 = tf.get_variable('b_conv4_3', [8])
        h_conv4_3 = tf.nn.relu(self.conv2d(h_conv3_3, w_conv4_3) + b_conv4_3)
        
        # merge ###########################################################
        h_conv4_merge = tf.concat([h_conv4_1, h_conv4_2, h_conv4_3], 3)
        
        w_conv5 = tf.get_variable('w_conv5', [1, 1, 30, 1])
        b_conv5 = tf.get_variable('b_conv5', [1])
        #h_conv5 = tf.nn.relu(self.conv2d(h_conv4_merge, w_conv5) + b_conv5)
        h_conv5 = self.conv2d(h_conv4_merge, w_conv5) + b_conv5
        
        y_pre = h_conv5

        return y_pre

    def test(self, sess):
        data = data_pre()
        
        mae = 0
        mse = 0
        
        for i in range(len(data)):
            if i % 20 == 0:
                print(i, '/', len(data))
            
            d = data[i]
            x_in = d[0]
            y_a = d[1]
            #print(x_in)
            #print(y_a)
            
            x_in = np.reshape(d[0], (1, d[0].shape[0], d[0].shape[1], 1))
            y_p = sess.run(self.pre_sum, feed_dict = {self.x: x_in})
            #print(y_a)
            #print(y_p)
            mae += abs(y_a - y_p)
            mse += (y_a - y_p) * (y_a - y_p)
            
        mae /= len(data)
        mse = math.sqrt(mse / len(data))
        print('mae: ', mae)
        print('mse: ', mse)
                            
net()









