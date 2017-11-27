from __future__ import division
import tensorflow as tf
import numpy as np
import random
import math
import argparse

LEARNING_RATE = 1e-4
BATCH_SIZE = 1
EPOCH = 500
START_NUM = 1
MAX_NUM = 40

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = '', help = 'input')
parser.add_argument('--model_dir', type = str, default = '', help = 'output')
flags, _ = parser.parse_known_args()

DEN_PATH = flags.data_dir + 'den/'
IMG_PATH = flags.data_dir + 'img/'

def loadnp(f):
    print('loadnp for ', f)
    with tf.gfile.Open(f, 'rb') as rf:
        content = rf.read()
    
    data = content.split('\n')
    data = [l.split() for l in data[:-1]]
    data = [list(map(float, l)) for l in data]
    data = [list(map(int, l)) for l in data]
    
    print(len(data))
    print(len(data[0]))
    n = np.array(data)
    print(n.shape)
    return n
    

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
            sess.run(tf.global_variables_initializer())
            self.train(sess)

            #saver = tf.train.Saver()
            #saver.save(sess, 'model/model.ckpt')

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

    def data_pre(self, img_num):
        print('data_pre function start.')
        den = loadnp(DEN_PATH + 'DEN_' + str(img_num) + '.txt')
        img = loadnp(IMG_PATH + 'IMG_' + str(img_num) + '.txt')

        height = img.shape[0]
        width = img.shape[1]

        x = np.array(img * 1.0 / 128.0, dtype = 'float32')
        x = x - 127.5
        x = np.reshape(x, (1, height, width, 1))
        
        
        ceilx = math.ceil(den.shape[0] / 4)
        ceily = math.ceil(den.shape[1] / 4)
        print('ceil: ', ceilx, 'x', ceily)
        print('intceil: ', int(ceilx), 'x', int(ceily))
        den_quarter = np.zeros((int(ceilx), int(ceily)))
        
        for i in range(den_quarter.shape[0]):
            for j in range(den_quarter.shape[1]):
                den_quarter[i, j] = np.sum(den[i * 4: i * 4 + 4, j * 4: j * 4 + 4])

        print('density shape: ', den.shape)
        print('density sum: ', np.sum(den))
        print('density quarter shape', den_quarter.shape)
        print('density quarter sum: ', np.sum(den_quarter))

        den_quarter = np.reshape(den_quarter, (1, den_quarter.shape[0], den_quarter.shape[1], 1))

        return x, den_quarter

    def train(self, sess):

        for epoch in range(EPOCH):
            print('***************************************************************************')
            print('epoch: ', epoch + 1)

            epoch_mae = 0

            for img_num in range(START_NUM, MAX_NUM + 1):
                print('*******************start epoch: ', epoch + 1, ', img_num: ', img_num)
                x_in, y_ground = self.data_pre(img_num)
                img = loadnp(IMG_PATH + 'IMG_' + str(img_num) + '.txt')
                print('image shape: ', img.shape)
                height = img.shape[0]
                width = img.shape[1]

                den_height = y_ground.shape[1]
                den_width = y_ground.shape[2]

                '''
                aaa_sum = 0
                mae_sum = 0
                for i in range(3):
                    for j in range(3):
                        x_h_len = math.ceil(height / 2)
                        x_w_len = math.ceil(width / 2)
                        y_h_len = math.ceil(den_height / 2)
                        y_w_len = math.ceil(den_width / 2)

                        x_cut = x_in[0, int(math.floor(height / 4 * i)): int(math.floor(height / 4 * i + x_h_len)), int(math.floor(width / 4 * j)): int(math.floor(width / 4 * j  + x_w_len)), 0]
                        x_cut = np.reshape(x_cut, (1, x_cut.shape[0], x_cut.shape[1], 1))
                        y_cut = y_ground[0, int(math.floor(den_height / 4 * i)): int(math.floor(den_height / 4 * i + y_h_len)), int(math.floor(den_width / 4 * j)): int(math.floor(den_width / 4 * j + y_w_len)), 0]
                        y_cut = np.reshape(y_cut, (1, y_cut.shape[0], y_cut.shape[1], 1))

                        _, l, y_a, y_p, act_s, pre_s, m, y_out = sess.run([self.train_step, self.loss, self.y_act, self.y_pre, \
                            self.act_sum, self.pre_sum, self.MAE, self.y_pre], \
                            feed_dict = {self.x: x_cut, self.y_act: y_cut})

                        y_a = np.array(y_a)
                        y_p = np.array(y_p)
                        #cha = y_a - y_p
                        #pingfang = cha ** 2
                        #he = np.sum(pingfang) / (pingfang.shape[0] * pingfang.shape[1])

                        #y_a.dtype = 'float32'
                        #y_p.dtype = 'float32'

                        #np.save('a' + str(i) + str(j) + '.npy', y_a)
                        #np.save('p' + str(i) + str(j) + '.npy', y_p)

                        print('loss: ', l)
                        #print('check loss: ', he)
                        print('act sum: ', act_s)
                        print('pre: ', pre_s)
                        #print('mae: ', m)

                        if i != 1 and j != 1:
                            aaa_sum += act_s

                        mae_sum += m
                #print('whole image act: ', aaa_sum)
                print('whole image mae: ', mae_sum)
                s = 'image num: ' + str(img_num) + ', mae: ' + str(mae_sum) + '\n'
                
                '''
                
                _, l, y_a, y_p, act_s, pre_s, m = sess.run([self.train_step, self.loss, self.y_act, self.y_pre, \
                    self.act_sum, self.pre_sum, self.MAE], \
                    feed_dict = {self.x: x_in, self.y_act: y_ground})
                print('loss: ', l)
                print('act sum: ', act_s)
                print('pre sum: ', pre_s)
                
                
                epoch_mae += m
            epoch_mae /= (MAX_NUM - START_NUM + 1)
            print('/////////////////////////////////epoch mae: ', epoch_mae)
            s = '*******************************************\nepoch num: ' + str(epoch + 1) + ', mae: ' + str(epoch_mae) + '\n'


a = net()
