import tensorflow as tf 
import numpy as np
import random
import cv2
import math

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
EPOCH = 2
MAX_NUM = 2

IMG_PATH = './shanghaitech/part_A_final/train_data/images/'
DEN_PATH = './density/part_A_final/train_data/'

class net:
    def __init__(self):
        print('init class...')
        
        
    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    def inf(self, x, h, w):
        x = tf.reshape(x, [1, h, w, 1])
        
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
        
        y_pre = tf.reshape(h_conv5, [math.ceil(h / 4), math.ceil(w / 4)])
        
        return y_pre

    def data_pre(self, img_num):
        print('start')
        den = np.loadtxt(DEN_PATH + 'DEN_' + str(img_num) + '.txt')
        print('load')
        img = cv2.imread(IMG_PATH + 'IMG_' + str(img_num) + '.jpg', 0)
        height = img.shape[0]
        width = img.shape[1]
        sub_height = math.floor(height / 2)
        sub_width = math.floor(width / 2)
            
        x = np.array(img * 1.0 / 255.0, dtype = 'float32')
        #x = x.reshape([1, height, width, 1])
        
        y_act = den
        
        return x, y_act
        
        #print('start')
        #y = inf(x)
        #print(y)

    def train(self):

        x_in, y_ground = self.data_pre(1)
        img = cv2.imread(IMG_PATH + 'IMG_' + '1' + '.jpg', 0)
        height = img.shape[0]
        width = img.shape[1]
        
        y_ground = np.resize(y_ground, (math.ceil(height / 4), math.ceil(width / 4)))
        
        x = tf.placeholder(tf.float32, [None, None])
        y_act = tf.placeholder(tf.float32, [None, None])
        y_pre = self.inf(x, height, width)
        
        loss = tf.reduce_mean((y_act - y_pre) ** 2)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
        
        MAE = tf.reduce_mean(np.abs(y_act - y_pre))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #for i in range(1):
                
                
            _, l, m, y_out = sess.run([train_step, loss, MAE, y_pre], \
                    feed_dict = {x: x_in, y_act: y_ground})
                
            #print(i)
                #print('y_out shape: ', y_out.shape)
                #print(y_out[0])
                #print('y_ground shape: ', y_ground.shape)
                #print(y_ground[0])
            print('loss: ', l)
            print('mae: ', m)
    
a = net()
a.train()
        
    


