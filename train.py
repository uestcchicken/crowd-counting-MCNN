import numpy as np 
import cv2
import tensorflow as tf
import os
import random
import math
import sys

LEARNING_RATE = 1e-5
EPOCH = 50000

if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 test.py A(or B)')
    exit()
print('dataset:', dataset)

train_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train/'
train_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den/'
val_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val/'
val_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val_den/'


def data_pre_train():
    print('loading data from dataset ', dataset, '...')
    train_img_names = os.listdir(train_path)
    img_num = len(train_img_names)

    train_data = []
    for i in range(img_num):
        if i % 100 == 0:
            print(i, '/', img_num)
        name = train_img_names[i]
        #print(name + '****************************')
        img = cv2.imread(train_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        #print(img.shape)
        den = np.loadtxt(open(train_den_path + name[:-4] + '.csv'), delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        #print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        train_data.append([img, den_quarter])

    print('load data finished.')
    return train_data
    
def data_pre_val():
    print('loading data for validation...')
    val_img_names = os.listdir(val_path)
    img_num = len(val_img_names)

    val_data = []
    for i in range(img_num):
        if i % 100 == 0:
            print(i, '/', img_num)
        name = val_img_names[i]
        #print(name + '****************************')
        img = cv2.imread(val_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        #print(img.shape)
        den = np.loadtxt(open(val_den_path + name[:-4] + '.csv'), delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        #print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        val_data.append([img, den_quarter])

    print('load data for validation finished.')
    return val_data

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
            #sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, 'model' + dataset + '/model.ckpt')
            self.train(sess)

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
        
        w_conv5 = tf.get_variable('w_conv5', [1, 1, 10, 1])
        b_conv5 = tf.get_variable('b_conv5', [1])
        #h_conv5 = tf.nn.relu(self.conv2d(h_conv4_merge, w_conv5) + b_conv5)
        #h_conv5 = self.conv2d(h_conv4_merge, w_conv5) + b_conv5
        h_conv5 = self.conv2d(h_conv4_2, w_conv5) + b_conv5
        
        y_pre = h_conv5

        return y_pre

    def train(self, sess):
        data_train = data_pre_train()
        data_val = data_pre_val()
        
        best_mae = 10000
        
        for epoch in range(EPOCH):
            #print('***************************************************************************')
            #print('epoch: ', epoch + 1)
            
            #training process
            epoch_mae = 0
            random.shuffle(data_train)
            for i in range(len(data_train)):
                data = data_train[i]
                x_in = np.reshape(data[0], (1, data[0].shape[0], data[0].shape[1], 1))
                #print(x_in.shape)
                y_ground = np.reshape(data[1], (1, data[1].shape[0], data[1].shape[1], 1))
                #print(y_ground.shape)    
                _, l, y_a, y_p, act_s, pre_s, m = sess.run( \
                    [self.train_step, self.loss, self.y_act, self.y_pre, \
                    self.act_sum, self.pre_sum, self.MAE], \
                    feed_dict = {self.x: x_in, self.y_act: y_ground})
                if i % 500 == 0:        
                    #print('loss: ', l)
                    #print('act sum: ', act_s)
                    #print('pre: ', pre_s)
                    print('epoch', epoch, 'step', i, 'mae:', m)
                epoch_mae += m
            epoch_mae /= len(data_train)
            print('epoch', epoch, 'train_mae:', epoch_mae)
            
            #validation process
            val_mae = 0
            val_mse = 0
            for i in range(len(data_val)):
                data = data_val[i]
                x_in = np.reshape(data[0], (1, data[0].shape[0], data[0].shape[1], 1))
                #print(x_in.shape)
                y_ground = np.reshape(data[1], (1, data[1].shape[0], data[1].shape[1], 1))
                #print(y_ground.shape)    
                act_s, pre_s, m = sess.run( \
                    [self.act_sum, self.pre_sum, self.MAE], \
                    feed_dict = {self.x: x_in, self.y_act: y_ground})
                #if i % 200 == 0:        
                    #print('loss: ', l)
                    #print('act sum: ', act_s)
                    #print('pre: ', pre_s)
                    #print('mae: ', m)
                val_mae += m
                val_mse += (act_s - pre_s) * (act_s - pre_s)
            val_mae /= len(data_val)
            val_mse = math.sqrt(val_mse / len(data_val))
            print('epoch', epoch, 'valid_mae:', val_mae, 'valid_mse:', val_mse)
            
            if val_mae < best_mae:
                best_mae = val_mae
                print('best mae so far, saving model.')
                saver = tf.train.Saver()
                saver.save(sess, 'model' + dataset + '/model.ckpt')
            else:
                print('best mae:', best_mae)
            print('**************************')
                            
net()









